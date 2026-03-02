/// HIR lowering pass: AST + TypeInfo -> HirProgram.
///
/// Desugars high-level constructs into simpler HIR nodes:
/// - Compound assignments -> read + op + write
/// - String interpolation -> runtime concat chain
/// - Pipelines -> function calls
/// - Filter/reduce -> runtime calls
/// - Nil coalesce -> if-nil-then-else
/// - Optional chain -> if-nil-then-nil-else-member
/// - Error propagation -> tuple extract + conditional return
/// - Destructuring -> individual bindings via index/member
/// - Loop-as-expression (collect) -> push-to-array loop
/// - Spread in arrays -> array concat
/// - Match -> if-else chain
use tok_parser::ast::{
    self, BinOp, Expr, FuncBody, InterpPart, LoopBody, LoopClause, MapKey, MatchBody, Param,
    Pattern, Program, SelectArm, Stmt, UnaryOp,
};
use tok_types::{Type, TypeInfo};

use crate::hir::*;

mod stmt;
mod expr;
mod desugar;
mod patterns;

// ═══════════════════════════════════════════════════════════════
// Lowerer state
// ═══════════════════════════════════════════════════════════════

struct Lowerer<'a> {
    type_info: &'a TypeInfo,
    tmp_counter: u32,
    /// Local variable type scopes (for function params and local vars).
    scopes: Vec<std::collections::HashMap<String, Type>>,
    /// Default parameter expressions for named functions (pre-collected for forward refs).
    /// Maps function name → vec of Option<default_expr> for each param.
    func_defaults: std::collections::HashMap<String, Vec<Option<Expr>>>,
    /// Variadic function info (pre-collected for forward refs).
    /// Maps function name → number of fixed (non-variadic) params.
    func_variadic: std::collections::HashMap<String, usize>,
    /// Total parameter count for named functions (pre-collected for spread desugaring).
    func_arity: std::collections::HashMap<String, usize>,
    /// Current method's self variable name (set during method lambda lowering).
    self_param: Option<String>,
    /// Tracks which variables have method fields (for self-injection at call sites).
    /// Maps variable name → set of field names that are methods.
    method_fields: std::collections::HashMap<String, std::collections::HashSet<String>>,
}

impl<'a> Lowerer<'a> {
    fn new(type_info: &'a TypeInfo) -> Self {
        Lowerer {
            type_info,
            tmp_counter: 0,
            scopes: Vec::new(),
            func_defaults: std::collections::HashMap::new(),
            func_variadic: std::collections::HashMap::new(),
            func_arity: std::collections::HashMap::new(),
            self_param: None,
            method_fields: std::collections::HashMap::new(),
        }
    }

    pub(super) fn push_scope(&mut self) {
        self.scopes.push(std::collections::HashMap::new());
    }

    pub(super) fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub(super) fn define_local(&mut self, name: &str, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn lookup_local(&self, name: &str) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    /// Generate a unique temporary variable name.
    pub(super) fn gensym(&mut self) -> String {
        self.tmp_counter += 1;
        format!("_tmp{}", self.tmp_counter)
    }

    // ═══════════════════════════════════════════════════════════
    // Type helpers
    // ═══════════════════════════════════════════════════════════

    /// Look up the type of a variable from local scopes, TypeInfo variables, or functions.
    pub(super) fn var_type(&self, name: &str) -> Type {
        // Check local scopes first (function params, loop vars, etc.)
        if let Some(ty) = self.lookup_local(name) {
            return ty.clone();
        }
        // Then check top-level variables
        if let Some(ty) = self.type_info.variables.get(name) {
            return ty.clone();
        }
        // Then check if it's a known function
        if let Some(ft) = self.type_info.functions.get(name) {
            return Type::Func(ft.clone());
        }
        Type::Any
    }

    /// Look up the return type of a named function.
    pub(super) fn func_ret_type(&self, name: &str) -> Type {
        self.type_info
            .functions
            .get(name)
            .map(|ft| *ft.ret.clone())
            .unwrap_or(Type::Any)
    }

    /// Infer the type of an AST expression (simplified, best-effort).
    pub(super) fn infer_expr_type(&self, expr: &Expr) -> Type {
        match expr {
            Expr::Int(_) => Type::Int,
            Expr::Float(_) => Type::Float,
            Expr::Str(_) => Type::Str,
            Expr::Bool(_) => Type::Bool,
            Expr::Nil => Type::Nil,
            Expr::Interp(_) => Type::Str,
            Expr::Ident(name) => self.var_type(name),
            Expr::Array(elts) => {
                if elts.is_empty() {
                    Type::Array(Box::new(Type::Any))
                } else {
                    let elem = self.infer_expr_type(&elts[0]);
                    Type::Array(Box::new(elem))
                }
            }
            Expr::Map(_) => Type::Map(Box::new(Type::Any)),
            Expr::Tuple(elts) => {
                Type::Tuple(elts.iter().map(|e| self.infer_expr_type(e)).collect())
            }
            Expr::Range { .. } => Type::Range,
            Expr::BinOp { op, left, right } => {
                let lt = self.infer_expr_type(left);
                let rt = self.infer_expr_type(right);
                self.infer_binop_type(op, &lt, &rt)
            }
            Expr::UnaryOp { op, expr } => {
                let t = self.infer_expr_type(expr);
                match op {
                    UnaryOp::Neg => match t {
                        Type::Int => Type::Int,
                        Type::Float => Type::Float,
                        _ => Type::Any,
                    },
                    UnaryOp::Not => Type::Bool,
                }
            }
            Expr::Index { expr, index } => {
                let target = self.infer_expr_type(expr);
                match &target {
                    Type::Array(inner) => *inner.clone(),
                    Type::Map(inner) => *inner.clone(),
                    Type::Tuple(elts) => {
                        if let Expr::Int(i) = index.as_ref() {
                            elts.get(*i as usize).cloned().unwrap_or(Type::Any)
                        } else {
                            Type::Any
                        }
                    }
                    Type::Str => Type::Str,
                    _ => Type::Any,
                }
            }
            Expr::Member { expr, field } => {
                let target = self.infer_expr_type(expr);
                self.infer_member_type(&target, field)
            }
            Expr::OptionalChain { expr, field } => {
                let target = self.infer_expr_type(expr);
                match &target {
                    Type::Nil => Type::Nil,
                    Type::Optional(inner) => {
                        let field_ty = self.infer_member_type(inner, field);
                        Type::Optional(Box::new(field_ty))
                    }
                    _ => {
                        let field_ty = self.infer_member_type(&target, field);
                        Type::Optional(Box::new(field_ty))
                    }
                }
            }
            Expr::Call { func, .. } => {
                if let Expr::Ident(name) = func.as_ref() {
                    self.func_ret_type(name)
                } else {
                    Type::Any
                }
            }
            Expr::Lambda {
                params, ret_type, ..
            } => {
                let param_types: Vec<tok_types::ParamType> = params
                    .iter()
                    .map(|p| tok_types::ParamType {
                        ty: Type::Any,
                        has_default: p.default.is_some(),
                    })
                    .collect();
                let ret = ret_type.as_ref().map(|_| Type::Any).unwrap_or(Type::Any);
                Type::Func(tok_types::FuncType {
                    params: param_types,
                    ret: Box::new(ret),
                    variadic: params.last().is_some_and(|p| p.variadic),
                })
            }
            Expr::Ternary {
                then_expr,
                else_expr,
                ..
            } => {
                let then_ty = self.infer_expr_type(then_expr);
                if let Some(else_e) = else_expr {
                    let else_ty = self.infer_expr_type(else_e);
                    tok_types::unify(&then_ty, &else_ty)
                } else {
                    then_ty
                }
            }
            Expr::Match { .. } => Type::Any,
            Expr::Loop { body, .. } => match body.as_ref() {
                LoopBody::Collect(e) => Type::Array(Box::new(self.infer_expr_type(e))),
                LoopBody::Block(_) => Type::Nil,
            },
            Expr::Block(_) => Type::Any,
            Expr::Pipeline { right, .. } => {
                let rt = self.infer_expr_type(right);
                match &rt {
                    Type::Func(ft) => *ft.ret.clone(),
                    _ => Type::Any,
                }
            }
            Expr::Filter { expr, .. } => self.infer_expr_type(expr),
            Expr::Reduce { init, expr, .. } => {
                if let Some(init_e) = init {
                    self.infer_expr_type(init_e)
                } else {
                    let arr_ty = self.infer_expr_type(expr);
                    match &arr_ty {
                        Type::Array(inner) => *inner.clone(),
                        _ => Type::Any,
                    }
                }
            }
            Expr::Spread(inner) => self.infer_expr_type(inner),
            Expr::Length(_) => Type::Int,
            Expr::NilCoalesce { left, right } => {
                let lt = self.infer_expr_type(left);
                let rt = self.infer_expr_type(right);
                match &lt {
                    Type::Optional(inner) => tok_types::unify(inner, &rt),
                    Type::Nil => rt,
                    _ => tok_types::unify(&lt, &rt),
                }
            }
            Expr::ErrorPropagate(inner) => {
                let inner_ty = self.infer_expr_type(inner);
                match &inner_ty {
                    Type::Result(ok_ty) => *ok_ty.clone(),
                    Type::Tuple(elts) if elts.len() == 2 => elts[0].clone(),
                    _ => Type::Any,
                }
            }
            Expr::ConditionalReturn { .. } => Type::Never,
            Expr::Go(body) => {
                let body_ty = self.infer_expr_type(body);
                Type::Handle(Box::new(body_ty))
            }
            Expr::Receive(inner) => {
                let inner_ty = self.infer_expr_type(inner);
                match &inner_ty {
                    Type::Channel(t) => *t.clone(),
                    Type::Handle(t) => *t.clone(),
                    _ => Type::Any,
                }
            }
            Expr::Send { .. } => Type::Nil,
            Expr::Select(_) => Type::Any,
            Expr::ImplicitSelf(_) => Type::Any,
            Expr::ProtoInit { .. } => Type::Map(Box::new(Type::Any)),
            Expr::Import(_) => Type::Map(Box::new(Type::Any)),
            Expr::Return(_) | Expr::Break | Expr::Continue => Type::Never,
        }
    }

    pub(super) fn infer_binop_type(&self, op: &BinOp, lt: &Type, rt: &Type) -> Type {
        tok_types::infer_binop_type(op, lt, rt)
    }

    pub(super) fn infer_member_type(&self, target_ty: &Type, field: &str) -> Type {
        tok_types::infer_member_type(target_ty, field)
    }

    // ═══════════════════════════════════════════════════════════
    // BinOp conversion
    // ═══════════════════════════════════════════════════════════

    pub(super) fn lower_binop(&self, op: &BinOp) -> HirBinOp {
        match op {
            BinOp::Add => HirBinOp::Add,
            BinOp::Sub => HirBinOp::Sub,
            BinOp::Mul => HirBinOp::Mul,
            BinOp::Div => HirBinOp::Div,
            BinOp::Mod => HirBinOp::Mod,
            BinOp::Pow => HirBinOp::Pow,
            BinOp::Eq => HirBinOp::Eq,
            BinOp::Neq => HirBinOp::Neq,
            BinOp::Lt => HirBinOp::Lt,
            BinOp::Gt => HirBinOp::Gt,
            BinOp::LtEq => HirBinOp::LtEq,
            BinOp::GtEq => HirBinOp::GtEq,
            BinOp::And => HirBinOp::And,
            BinOp::Or => HirBinOp::Or,
            BinOp::BitAnd => HirBinOp::BitAnd,
            BinOp::BitOr => HirBinOp::BitOr,
            BinOp::BitXor => HirBinOp::BitXor,
            BinOp::Append => unreachable!("Append is desugared to RuntimeCall"),
            BinOp::Shr => HirBinOp::Shr,
        }
    }

    pub(super) fn lower_unaryop(&self, op: &UnaryOp) -> HirUnaryOp {
        match op {
            UnaryOp::Neg => HirUnaryOp::Neg,
            UnaryOp::Not => HirUnaryOp::Not,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════

/// Lower a parsed + type-checked program to HIR.
///
/// Takes the AST and type information and produces a simplified HIR
/// where all syntactic sugar has been desugared into primitive operations.
pub fn lower(program: &Program, type_info: &TypeInfo) -> HirProgram {
    let mut lowerer = Lowerer::new(type_info);
    // Pre-collect default parameter expressions for named functions.
    // This handles forward references (call before declaration).
    for stmt in program {
        if let Stmt::FuncDecl { name, params, .. } = stmt {
            if params.iter().any(|p| p.default.is_some()) {
                lowerer.func_defaults.insert(
                    name.clone(),
                    params.iter().map(|p| p.default.clone()).collect(),
                );
            }
            if params.last().is_some_and(|p| p.variadic) {
                lowerer.func_variadic.insert(name.clone(), params.len() - 1);
            }
            lowerer.func_arity.insert(name.clone(), params.len());
        }
    }
    lowerer.push_scope(); // top-level scope for local variable tracking
    let result = lowerer.lower_program(program);
    lowerer.pop_scope();
    result
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use tok_parser::ast::{BinOp, Expr, FuncBody, InterpPart, Param, Stmt};

    fn lower_program(stmts: Vec<Stmt>) -> HirProgram {
        let ti = tok_types::check(&stmts);
        lower(&stmts, &ti)
    }

    // ─── Test 1: Simple assignment (no desugaring) ─────────────

    #[test]
    fn simple_assignment() {
        let prog = vec![Stmt::Assign {
            name: "x".into(),
            ty: None,
            value: Expr::Int(42),
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Assign { name, value, .. } => {
                assert_eq!(name, "x");
                assert!(matches!(value.kind, HirExprKind::Int(42)));
                assert!(matches!(value.ty, Type::Int));
            }
            _ => panic!("expected Assign"),
        }
    }

    // ─── Test 2: Compound assignment desugaring ────────────────

    #[test]
    fn compound_assignment_desugared() {
        let prog = vec![
            Stmt::Assign {
                name: "x".into(),
                ty: None,
                value: Expr::Int(10),
            },
            Stmt::CompoundAssign {
                name: "x".into(),
                op: BinOp::Add,
                value: Expr::Int(1),
            },
        ];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 2);
        match &hir[1] {
            HirStmt::Assign { name, value, .. } => {
                assert_eq!(name, "x");
                match &value.kind {
                    HirExprKind::BinOp { op, left, right } => {
                        assert!(matches!(op, HirBinOp::Add));
                        assert!(matches!(left.kind, HirExprKind::Ident(ref n) if n == "x"));
                        assert!(matches!(right.kind, HirExprKind::Int(1)));
                    }
                    _ => panic!("expected BinOp"),
                }
            }
            _ => panic!("expected Assign"),
        }
    }

    // ─── Test 3: String interpolation desugaring ───────────────

    #[test]
    fn string_interpolation_desugared() {
        let prog = vec![Stmt::Assign {
            name: "s".into(),
            ty: None,
            value: Expr::Interp(vec![
                InterpPart::Lit("hello ".into()),
                InterpPart::Expr(Expr::Ident("name".into())),
                InterpPart::Lit("!".into()),
            ]),
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Assign { name, value, .. } => {
                assert_eq!(name, "s");
                assert!(matches!(value.ty, Type::Str));
                // Should be a chain of tok_string_concat calls
                match &value.kind {
                    HirExprKind::RuntimeCall { name, args } => {
                        assert_eq!(name, "tok_string_concat");
                        assert_eq!(args.len(), 2);
                        // left should be another concat
                        match &args[0].kind {
                            HirExprKind::RuntimeCall {
                                name: inner_name,
                                args: inner_args,
                            } => {
                                assert_eq!(inner_name, "tok_string_concat");
                                assert_eq!(inner_args.len(), 2);
                                assert!(matches!(
                                    inner_args[0].kind,
                                    HirExprKind::Str(ref s) if s == "hello "
                                ));
                                // inner_args[1] should be a value_to_string or ident
                                match &inner_args[1].kind {
                                    HirExprKind::RuntimeCall {
                                        name: conv_name, ..
                                    } => {
                                        assert!(
                                            conv_name == "tok_value_to_string"
                                                || conv_name == "tok_int_to_string"
                                                || conv_name == "tok_float_to_string"
                                                || conv_name == "tok_bool_to_string"
                                        );
                                    }
                                    HirExprKind::Ident(_) => {
                                        // If type is known to be Str, it's used directly
                                    }
                                    other => panic!(
                                        "expected RuntimeCall or Ident for interp expr, got {:?}",
                                        other
                                    ),
                                }
                            }
                            _ => panic!("expected inner RuntimeCall"),
                        }
                        // right should be "!"
                        assert!(matches!(args[1].kind, HirExprKind::Str(ref s) if s == "!"));
                    }
                    _ => panic!("expected RuntimeCall for interp, got {:?}", value.kind),
                }
            }
            _ => panic!("expected Assign"),
        }
    }

    // ─── Test 4: Pipeline desugaring ───────────────────────────

    #[test]
    fn pipeline_desugared_simple() {
        // x |> f -> f(x)
        let prog = vec![
            Stmt::Assign {
                name: "x".into(),
                ty: None,
                value: Expr::Int(42),
            },
            Stmt::Expr(Expr::Pipeline {
                left: Box::new(Expr::Ident("x".into())),
                right: Box::new(Expr::Ident("f".into())),
            }),
        ];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 2);
        match &hir[1] {
            HirStmt::Expr(expr) => match &expr.kind {
                HirExprKind::Call { func, args } => {
                    assert!(matches!(func.kind, HirExprKind::Ident(ref n) if n == "f"));
                    assert_eq!(args.len(), 1);
                    assert!(matches!(args[0].kind, HirExprKind::Ident(ref n) if n == "x"));
                }
                _ => panic!("expected Call"),
            },
            _ => panic!("expected Expr stmt"),
        }
    }

    #[test]
    fn pipeline_desugared_with_args() {
        // x |> f(y) -> f(x, y)
        let prog = vec![Stmt::Expr(Expr::Pipeline {
            left: Box::new(Expr::Int(1)),
            right: Box::new(Expr::Call {
                func: Box::new(Expr::Ident("f".into())),
                args: vec![Expr::Int(2)],
            }),
        })];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Expr(expr) => match &expr.kind {
                HirExprKind::Call { func, args } => {
                    assert!(matches!(func.kind, HirExprKind::Ident(ref n) if n == "f"));
                    assert_eq!(args.len(), 2);
                    assert!(matches!(args[0].kind, HirExprKind::Int(1)));
                    assert!(matches!(args[1].kind, HirExprKind::Int(2)));
                }
                _ => panic!("expected Call"),
            },
            _ => panic!("expected Expr stmt"),
        }
    }

    // ─── Test 5: Tuple destructure desugaring ──────────────────

    #[test]
    fn tuple_destructure_desugared() {
        // a b = (1, "hi")
        let prog = vec![Stmt::TupleDestructure {
            names: vec!["a".into(), "b".into()],
            value: Expr::Tuple(vec![Expr::Int(1), Expr::Str("hi".into())]),
        }];
        let hir = lower_program(prog);
        // Should produce: _tmp = (1, "hi"); a = _tmp[0]; b = _tmp[1]
        assert_eq!(hir.len(), 3);

        // First: _tmp = (1, "hi")
        match &hir[0] {
            HirStmt::Assign { name, value, .. } => {
                assert!(name.starts_with("_tmp"));
                assert!(matches!(value.kind, HirExprKind::Tuple(_)));
            }
            _ => panic!("expected Assign for tmp"),
        }

        // Second: a = _tmp[0]
        match &hir[1] {
            HirStmt::Assign { name, value, .. } => {
                assert_eq!(name, "a");
                match &value.kind {
                    HirExprKind::Index { index, .. } => {
                        assert!(matches!(index.kind, HirExprKind::Int(0)));
                    }
                    _ => panic!("expected Index"),
                }
            }
            _ => panic!("expected Assign"),
        }

        // Third: b = _tmp[1]
        match &hir[2] {
            HirStmt::Assign { name, value, .. } => {
                assert_eq!(name, "b");
                match &value.kind {
                    HirExprKind::Index { index, .. } => {
                        assert!(matches!(index.kind, HirExprKind::Int(1)));
                    }
                    _ => panic!("expected Index"),
                }
            }
            _ => panic!("expected Assign"),
        }
    }

    // ─── Test 6: Match desugaring ──────────────────────────────

    #[test]
    fn match_desugared_to_if_chain() {
        // x ?= { 1: "one"; 2: "two"; _: "other" }
        let prog = vec![Stmt::Assign {
            name: "result".into(),
            ty: None,
            value: Expr::Match {
                subject: Some(Box::new(Expr::Ident("x".into()))),
                arms: vec![
                    ast::MatchArm {
                        pattern: Pattern::Int(1),
                        body: MatchBody::Expr(Expr::Str("one".into())),
                    },
                    ast::MatchArm {
                        pattern: Pattern::Int(2),
                        body: MatchBody::Expr(Expr::Str("two".into())),
                    },
                    ast::MatchArm {
                        pattern: Pattern::Wildcard,
                        body: MatchBody::Expr(Expr::Str("other".into())),
                    },
                ],
            },
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);

        match &hir[0] {
            HirStmt::Assign { name, value, .. } => {
                assert_eq!(name, "result");
                // Should be a block containing tmp assignment + if-else chain
                match &value.kind {
                    HirExprKind::Block { stmts, expr } => {
                        // One stmt: _tmp = x
                        assert_eq!(stmts.len(), 1);
                        // expr is the if-else chain
                        let if_expr = expr.as_ref().unwrap();
                        match &if_expr.kind {
                            HirExprKind::If {
                                cond,
                                then_expr,
                                else_expr,
                                ..
                            } => {
                                // cond: _tmp == 1
                                assert!(matches!(cond.kind, HirExprKind::BinOp { .. }));
                                // then: "one"
                                let then_e = then_expr.as_ref().unwrap();
                                assert!(matches!(&then_e.kind, HirExprKind::Str(s) if s == "one"));
                                // else: another if
                                let else_e = else_expr.as_ref().unwrap();
                                match &else_e.kind {
                                    HirExprKind::If {
                                        then_expr: inner_then,
                                        else_expr: inner_else,
                                        ..
                                    } => {
                                        let inner_then_e = inner_then.as_ref().unwrap();
                                        assert!(matches!(
                                            &inner_then_e.kind,
                                            HirExprKind::Str(s) if s == "two"
                                        ));
                                        let inner_else_e = inner_else.as_ref().unwrap();
                                        assert!(matches!(
                                            &inner_else_e.kind,
                                            HirExprKind::Str(s) if s == "other"
                                        ));
                                    }
                                    _ => panic!("expected nested If"),
                                }
                            }
                            _ => panic!("expected If"),
                        }
                    }
                    _ => panic!("expected Block for match"),
                }
            }
            _ => panic!("expected Assign"),
        }
    }

    // ─── Test 7: Filter and reduce desugaring ──────────────────

    #[test]
    fn filter_desugared_to_runtime_call() {
        let prog = vec![Stmt::Expr(Expr::Filter {
            expr: Box::new(Expr::Ident("arr".into())),
            pred: Box::new(Expr::Ident("pred".into())),
        })];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Expr(expr) => match &expr.kind {
                HirExprKind::RuntimeCall { name, args } => {
                    assert_eq!(name, "tok_array_filter");
                    assert_eq!(args.len(), 2);
                }
                _ => panic!("expected RuntimeCall"),
            },
            _ => panic!("expected Expr stmt"),
        }
    }

    #[test]
    fn reduce_desugared_to_runtime_call() {
        let prog = vec![Stmt::Expr(Expr::Reduce {
            expr: Box::new(Expr::Ident("arr".into())),
            init: Some(Box::new(Expr::Int(0))),
            func: Box::new(Expr::Ident("add".into())),
        })];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Expr(expr) => match &expr.kind {
                HirExprKind::RuntimeCall { name, args } => {
                    assert_eq!(name, "tok_array_reduce");
                    assert_eq!(args.len(), 3);
                    // args: arr, init(0), func
                    assert!(matches!(args[1].kind, HirExprKind::Int(0)));
                }
                _ => panic!("expected RuntimeCall"),
            },
            _ => panic!("expected Expr stmt"),
        }
    }

    // ─── Additional desugaring tests ───────────────────────────

    #[test]
    fn nil_coalesce_desugared() {
        let prog = vec![Stmt::Assign {
            name: "y".into(),
            ty: None,
            value: Expr::NilCoalesce {
                left: Box::new(Expr::Ident("x".into())),
                right: Box::new(Expr::Int(42)),
            },
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Block { stmts, expr } => {
                    assert_eq!(stmts.len(), 1);
                    let if_expr = expr.as_ref().unwrap();
                    assert!(matches!(if_expr.kind, HirExprKind::If { .. }));
                }
                _ => panic!("expected Block for nil coalesce"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn error_propagation_desugared() {
        let prog = vec![Stmt::Expr(Expr::ErrorPropagate(Box::new(Expr::Ident(
            "result".into(),
        ))))];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Expr(expr) => match &expr.kind {
                HirExprKind::Block { stmts, expr } => {
                    assert_eq!(stmts.len(), 1); // _tmp = result
                    let if_expr = expr.as_ref().unwrap();
                    match &if_expr.kind {
                        HirExprKind::If {
                            then_body,
                            else_expr,
                            ..
                        } => {
                            // then_body has a Return
                            assert!(matches!(then_body[0], HirStmt::Return(_)));
                            // else_expr extracts the ok value
                            assert!(else_expr.is_some());
                        }
                        _ => panic!("expected If"),
                    }
                }
                _ => panic!("expected Block"),
            },
            _ => panic!("expected Expr stmt"),
        }
    }

    #[test]
    fn conditional_return_desugared() {
        let prog = vec![Stmt::Expr(Expr::ConditionalReturn {
            cond: Box::new(Expr::Bool(true)),
            value: Box::new(Expr::Int(99)),
        })];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Expr(expr) => match &expr.kind {
                HirExprKind::If {
                    cond, then_body, ..
                } => {
                    assert!(matches!(cond.kind, HirExprKind::Bool(true)));
                    assert!(matches!(then_body[0], HirStmt::Return(Some(_))));
                }
                _ => panic!("expected If"),
            },
            _ => panic!("expected Expr stmt"),
        }
    }

    #[test]
    fn compound_index_assign_desugared() {
        let prog = vec![Stmt::CompoundIndexAssign {
            target: Expr::Ident("arr".into()),
            index: Expr::Int(0),
            op: BinOp::Add,
            value: Expr::Int(1),
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::IndexAssign { value, .. } => match &value.kind {
                HirExprKind::BinOp { op, left, right } => {
                    assert!(matches!(op, HirBinOp::Add));
                    assert!(matches!(left.kind, HirExprKind::Index { .. }));
                    assert!(matches!(right.kind, HirExprKind::Int(1)));
                }
                _ => panic!("expected BinOp"),
            },
            _ => panic!("expected IndexAssign"),
        }
    }

    #[test]
    fn compound_member_assign_desugared() {
        let prog = vec![Stmt::CompoundMemberAssign {
            target: Expr::Ident("m".into()),
            field: "x".into(),
            op: BinOp::Mul,
            value: Expr::Int(2),
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::MemberAssign { field, value, .. } => {
                assert_eq!(field, "x");
                match &value.kind {
                    HirExprKind::BinOp { op, left, right } => {
                        assert!(matches!(op, HirBinOp::Mul));
                        assert!(matches!(left.kind, HirExprKind::Member { .. }));
                        assert!(matches!(right.kind, HirExprKind::Int(2)));
                    }
                    _ => panic!("expected BinOp"),
                }
            }
            _ => panic!("expected MemberAssign"),
        }
    }

    #[test]
    fn map_destructure_desugared() {
        let prog = vec![Stmt::MapDestructure {
            names: vec!["a".into(), "b".into()],
            value: Expr::Ident("m".into()),
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 3); // _tmp = m; a = _tmp.a; b = _tmp.b
        match &hir[1] {
            HirStmt::Assign { name, value, .. } => {
                assert_eq!(name, "a");
                match &value.kind {
                    HirExprKind::Member { field, .. } => assert_eq!(field, "a"),
                    _ => panic!("expected Member"),
                }
            }
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn array_destructure_desugared() {
        let prog = vec![Stmt::ArrayDestructure {
            head: "h".into(),
            tail: "t".into(),
            value: Expr::Ident("arr".into()),
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 3); // _tmp = arr; h = _tmp[0]; t = slice(...)
        match &hir[1] {
            HirStmt::Assign { name, value, .. } => {
                assert_eq!(name, "h");
                assert!(matches!(value.kind, HirExprKind::Index { .. }));
            }
            _ => panic!("expected Assign for head"),
        }
        match &hir[2] {
            HirStmt::Assign { name, value, .. } => {
                assert_eq!(name, "t");
                match &value.kind {
                    HirExprKind::RuntimeCall { name, .. } => {
                        assert_eq!(name, "tok_array_slice");
                    }
                    _ => panic!("expected RuntimeCall for tail"),
                }
            }
            _ => panic!("expected Assign for tail"),
        }
    }

    #[test]
    fn spread_in_array_desugared() {
        let prog = vec![Stmt::Assign {
            name: "result".into(),
            ty: None,
            value: Expr::Array(vec![
                Expr::Spread(Box::new(Expr::Ident("a".into()))),
                Expr::Int(42),
            ]),
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Block { stmts, expr } => {
                    // stmts: alloc, concat(a), push(42)
                    assert_eq!(stmts.len(), 3);
                    assert!(expr.is_some());
                }
                _ => panic!("expected Block for spread array"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn func_decl_lowered() {
        let prog = vec![Stmt::FuncDecl {
            name: "add".into(),
            params: vec![
                Param {
                    name: "a".into(),
                    ty: None,
                    default: None,
                    variadic: false,
                },
                Param {
                    name: "b".into(),
                    ty: None,
                    default: None,
                    variadic: false,
                },
            ],
            ret_type: None,
            body: FuncBody::Expr(Box::new(Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::Ident("a".into())),
                right: Box::new(Expr::Ident("b".into())),
            })),
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::FuncDecl {
                name, params, body, ..
            } => {
                assert_eq!(name, "add");
                assert_eq!(params.len(), 2);
                // Expression body -> Return(expr)
                assert_eq!(body.len(), 1);
                assert!(matches!(body[0], HirStmt::Return(Some(_))));
            }
            _ => panic!("expected FuncDecl"),
        }
    }

    #[test]
    fn default_params_padded_at_call_site() {
        // f add(a b=10)=a+b
        // r=add(5)         -> should desugar to add(5, 10)
        let prog = vec![
            Stmt::FuncDecl {
                name: "add".into(),
                params: vec![
                    Param {
                        name: "a".into(),
                        ty: None,
                        default: None,
                        variadic: false,
                    },
                    Param {
                        name: "b".into(),
                        ty: None,
                        default: Some(Expr::Int(10)),
                        variadic: false,
                    },
                ],
                ret_type: None,
                body: FuncBody::Expr(Box::new(Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(Expr::Ident("a".into())),
                    right: Box::new(Expr::Ident("b".into())),
                })),
            },
            Stmt::Assign {
                name: "r".into(),
                ty: None,
                value: Expr::Call {
                    func: Box::new(Expr::Ident("add".into())),
                    args: vec![Expr::Int(5)],
                },
            },
        ];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 2);
        // The call add(5) should have been padded to add(5, 10)
        match &hir[1] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Call { args, .. } => {
                    assert_eq!(args.len(), 2, "default param should pad args to 2");
                    assert!(matches!(args[0].kind, HirExprKind::Int(5)));
                    assert!(matches!(args[1].kind, HirExprKind::Int(10)));
                }
                _ => panic!("expected Call"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn default_params_not_padded_when_provided() {
        // f add(a b=10)=a+b
        // r=add(5 20)       -> should stay as add(5, 20)
        let prog = vec![
            Stmt::FuncDecl {
                name: "add".into(),
                params: vec![
                    Param {
                        name: "a".into(),
                        ty: None,
                        default: None,
                        variadic: false,
                    },
                    Param {
                        name: "b".into(),
                        ty: None,
                        default: Some(Expr::Int(10)),
                        variadic: false,
                    },
                ],
                ret_type: None,
                body: FuncBody::Expr(Box::new(Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(Expr::Ident("a".into())),
                    right: Box::new(Expr::Ident("b".into())),
                })),
            },
            Stmt::Assign {
                name: "r".into(),
                ty: None,
                value: Expr::Call {
                    func: Box::new(Expr::Ident("add".into())),
                    args: vec![Expr::Int(5), Expr::Int(20)],
                },
            },
        ];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 2);
        match &hir[1] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Call { args, .. } => {
                    assert_eq!(args.len(), 2, "should not add extra args when all provided");
                    assert!(matches!(args[0].kind, HirExprKind::Int(5)));
                    assert!(matches!(args[1].kind, HirExprKind::Int(20)));
                }
                _ => panic!("expected Call"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn default_params_forward_reference() {
        // r=add(5)          -> call before declaration (forward reference)
        // f add(a b=10)=a+b
        let prog = vec![
            Stmt::Assign {
                name: "r".into(),
                ty: None,
                value: Expr::Call {
                    func: Box::new(Expr::Ident("add".into())),
                    args: vec![Expr::Int(5)],
                },
            },
            Stmt::FuncDecl {
                name: "add".into(),
                params: vec![
                    Param {
                        name: "a".into(),
                        ty: None,
                        default: None,
                        variadic: false,
                    },
                    Param {
                        name: "b".into(),
                        ty: None,
                        default: Some(Expr::Int(10)),
                        variadic: false,
                    },
                ],
                ret_type: None,
                body: FuncBody::Expr(Box::new(Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(Expr::Ident("a".into())),
                    right: Box::new(Expr::Ident("b".into())),
                })),
            },
        ];
        let hir = lower_program(prog);
        // Forward ref call should still be padded thanks to pre-scan
        match &hir[0] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Call { args, .. } => {
                    assert_eq!(args.len(), 2, "forward ref should be padded");
                    assert!(matches!(args[1].kind, HirExprKind::Int(10)));
                }
                _ => panic!("expected Call"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn optional_chain_desugared() {
        let prog = vec![Stmt::Expr(Expr::OptionalChain {
            expr: Box::new(Expr::Ident("x".into())),
            field: "name".into(),
        })];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Expr(expr) => match &expr.kind {
                HirExprKind::Block { stmts, expr } => {
                    assert_eq!(stmts.len(), 1); // _tmp = x
                    let if_expr = expr.as_ref().unwrap();
                    assert!(matches!(if_expr.kind, HirExprKind::If { .. }));
                }
                _ => panic!("expected Block for optional chain"),
            },
            _ => panic!("expected Expr stmt"),
        }
    }

    #[test]
    fn loop_collect_desugared() {
        let prog = vec![Stmt::Assign {
            name: "squares".into(),
            ty: None,
            value: Expr::Loop {
                clause: Box::new(LoopClause::ForRange {
                    var: "i".into(),
                    range: Expr::Range {
                        start: Box::new(Expr::Int(0)),
                        end: Box::new(Expr::Int(5)),
                        inclusive: false,
                    },
                }),
                body: Box::new(LoopBody::Collect(Expr::BinOp {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Ident("i".into())),
                    right: Box::new(Expr::Ident("i".into())),
                })),
            },
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Block { stmts, expr } => {
                    // stmts: _collect = alloc(); loop { _collect = push(...) }
                    assert_eq!(stmts.len(), 2);
                    assert!(expr.is_some());
                }
                _ => panic!("expected Block for collect loop"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn variadic_args_wrapped_into_array() {
        // f sum(..nums)=nums
        // r=sum(1 2 3)   -> should desugar to sum([1, 2, 3])
        let prog = vec![
            Stmt::FuncDecl {
                name: "sum".into(),
                params: vec![Param {
                    name: "nums".into(),
                    ty: None,
                    default: None,
                    variadic: true,
                }],
                ret_type: None,
                body: FuncBody::Expr(Box::new(Expr::Ident("nums".into()))),
            },
            Stmt::Assign {
                name: "r".into(),
                ty: None,
                value: Expr::Call {
                    func: Box::new(Expr::Ident("sum".into())),
                    args: vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)],
                },
            },
        ];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 2);
        match &hir[1] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Call { args, .. } => {
                    assert_eq!(
                        args.len(),
                        1,
                        "variadic args should be wrapped into 1 array arg"
                    );
                    match &args[0].kind {
                        HirExprKind::Array(elts) => {
                            assert_eq!(elts.len(), 3);
                            assert!(matches!(elts[0].kind, HirExprKind::Int(1)));
                            assert!(matches!(elts[1].kind, HirExprKind::Int(2)));
                            assert!(matches!(elts[2].kind, HirExprKind::Int(3)));
                        }
                        _ => panic!("expected Array wrapping variadic args"),
                    }
                }
                _ => panic!("expected Call"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn variadic_with_fixed_params() {
        // f foo(a b ..rest)=a
        // r=foo(1 2 3 4)   -> should desugar to foo(1, 2, [3, 4])
        let prog = vec![
            Stmt::FuncDecl {
                name: "foo".into(),
                params: vec![
                    Param {
                        name: "a".into(),
                        ty: None,
                        default: None,
                        variadic: false,
                    },
                    Param {
                        name: "b".into(),
                        ty: None,
                        default: None,
                        variadic: false,
                    },
                    Param {
                        name: "rest".into(),
                        ty: None,
                        default: None,
                        variadic: true,
                    },
                ],
                ret_type: None,
                body: FuncBody::Expr(Box::new(Expr::Ident("a".into()))),
            },
            Stmt::Assign {
                name: "r".into(),
                ty: None,
                value: Expr::Call {
                    func: Box::new(Expr::Ident("foo".into())),
                    args: vec![Expr::Int(1), Expr::Int(2), Expr::Int(3), Expr::Int(4)],
                },
            },
        ];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 2);
        match &hir[1] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Call { args, .. } => {
                    assert_eq!(args.len(), 3, "should be 2 fixed + 1 array");
                    assert!(matches!(args[0].kind, HirExprKind::Int(1)));
                    assert!(matches!(args[1].kind, HirExprKind::Int(2)));
                    match &args[2].kind {
                        HirExprKind::Array(elts) => {
                            assert_eq!(elts.len(), 2);
                            assert!(matches!(elts[0].kind, HirExprKind::Int(3)));
                            assert!(matches!(elts[1].kind, HirExprKind::Int(4)));
                        }
                        _ => panic!("expected Array for variadic rest"),
                    }
                }
                _ => panic!("expected Call"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn variadic_no_extra_args_gives_empty_array() {
        // f foo(a ..rest)=a
        // r=foo(1)   -> should desugar to foo(1, [])
        let prog = vec![
            Stmt::FuncDecl {
                name: "foo".into(),
                params: vec![
                    Param {
                        name: "a".into(),
                        ty: None,
                        default: None,
                        variadic: false,
                    },
                    Param {
                        name: "rest".into(),
                        ty: None,
                        default: None,
                        variadic: true,
                    },
                ],
                ret_type: None,
                body: FuncBody::Expr(Box::new(Expr::Ident("a".into()))),
            },
            Stmt::Assign {
                name: "r".into(),
                ty: None,
                value: Expr::Call {
                    func: Box::new(Expr::Ident("foo".into())),
                    args: vec![Expr::Int(1)],
                },
            },
        ];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 2);
        match &hir[1] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Call { args, .. } => {
                    assert_eq!(args.len(), 2, "should be 1 fixed + 1 empty array");
                    assert!(matches!(args[0].kind, HirExprKind::Int(1)));
                    match &args[1].kind {
                        HirExprKind::Array(elts) => {
                            assert_eq!(elts.len(), 0, "empty variadic should give empty array");
                        }
                        _ => panic!("expected empty Array for variadic rest"),
                    }
                }
                _ => panic!("expected Call"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn variadic_param_type_is_array() {
        // f sum(..nums)=nums  ->  param type should be Array(Any)
        let prog = vec![Stmt::FuncDecl {
            name: "sum".into(),
            params: vec![Param {
                name: "nums".into(),
                ty: None,
                default: None,
                variadic: true,
            }],
            ret_type: None,
            body: FuncBody::Expr(Box::new(Expr::Ident("nums".into()))),
        }];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 1);
        match &hir[0] {
            HirStmt::FuncDecl { params, .. } => {
                assert_eq!(params.len(), 1);
                assert!(params[0].variadic);
                assert!(
                    matches!(params[0].ty, Type::Array(_)),
                    "variadic param should have Array type"
                );
            }
            _ => panic!("expected FuncDecl"),
        }
    }

    #[test]
    fn spread_in_call_desugared_to_block() {
        // f add(a b)=a+b
        // r=add(..arr)   -> should desugar to block with array build + index extraction
        let prog = vec![
            Stmt::FuncDecl {
                name: "add".into(),
                params: vec![
                    Param {
                        name: "a".into(),
                        ty: None,
                        default: None,
                        variadic: false,
                    },
                    Param {
                        name: "b".into(),
                        ty: None,
                        default: None,
                        variadic: false,
                    },
                ],
                ret_type: None,
                body: FuncBody::Expr(Box::new(Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(Expr::Ident("a".into())),
                    right: Box::new(Expr::Ident("b".into())),
                })),
            },
            Stmt::Assign {
                name: "r".into(),
                ty: None,
                value: Expr::Call {
                    func: Box::new(Expr::Ident("add".into())),
                    args: vec![Expr::Spread(Box::new(Expr::Ident("arr".into())))],
                },
            },
        ];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 2);
        // The call should be wrapped in a Block (array build + call)
        match &hir[1] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Block { stmts, expr } => {
                    // stmts: alloc + concat (for the spread)
                    assert!(stmts.len() >= 2, "should have alloc + concat stmts");
                    // expr: the call with indexed args
                    let call = expr.as_ref().unwrap();
                    match &call.kind {
                        HirExprKind::Call { args, .. } => {
                            assert_eq!(args.len(), 2, "add takes 2 params, should extract 2 args");
                            // Both args should be Index expressions
                            assert!(
                                matches!(args[0].kind, HirExprKind::Index { .. }),
                                "first arg should be Index"
                            );
                            assert!(
                                matches!(args[1].kind, HirExprKind::Index { .. }),
                                "second arg should be Index"
                            );
                        }
                        _ => panic!("expected Call inside Block"),
                    }
                }
                _ => panic!("expected Block for spread call"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn spread_in_variadic_call_desugared() {
        // f sum(..nums)=nums
        // r=sum(1 ..arr 2)   -> should build flat array, extract via slice
        let prog = vec![
            Stmt::FuncDecl {
                name: "sum".into(),
                params: vec![Param {
                    name: "nums".into(),
                    ty: None,
                    default: None,
                    variadic: true,
                }],
                ret_type: None,
                body: FuncBody::Expr(Box::new(Expr::Ident("nums".into()))),
            },
            Stmt::Assign {
                name: "r".into(),
                ty: None,
                value: Expr::Call {
                    func: Box::new(Expr::Ident("sum".into())),
                    args: vec![
                        Expr::Int(1),
                        Expr::Spread(Box::new(Expr::Ident("arr".into()))),
                        Expr::Int(2),
                    ],
                },
            },
        ];
        let hir = lower_program(prog);
        assert_eq!(hir.len(), 2);
        // The call should be wrapped in a Block
        match &hir[1] {
            HirStmt::Assign { value, .. } => match &value.kind {
                HirExprKind::Block { stmts, expr } => {
                    // stmts: alloc + push(1) + concat(arr) + push(2) = 4
                    assert_eq!(stmts.len(), 4, "should have alloc + push + concat + push");
                    // expr: call with 1 arg (the sliced array for variadic)
                    let call = expr.as_ref().unwrap();
                    match &call.kind {
                        HirExprKind::Call { args, .. } => {
                            // 0 fixed params → slice from 0..len = the whole array
                            assert_eq!(args.len(), 1, "variadic with 0 fixed → 1 array arg");
                            assert!(
                                matches!(args[0].kind, HirExprKind::RuntimeCall { .. }),
                                "arg should be tok_array_slice"
                            );
                        }
                        _ => panic!("expected Call inside Block"),
                    }
                }
                _ => panic!("expected Block for spread call"),
            },
            _ => panic!("expected Assign"),
        }
    }
}
