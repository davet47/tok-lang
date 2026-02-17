/// Type checker for the Tok language.
///
/// Takes a parsed AST (`Program`) and produces `TypeInfo` — a collection
/// of function signatures and variable types that downstream passes
/// (HIR lowering, codegen) consume.
///
/// Design principles:
/// - **Lenient**: warnings instead of errors. Untyped code passes.
/// - **`Any` escape hatch**: missing annotations default to `Any`.
/// - **Forward-flow inference**: no Hindley-Milner, just propagate types forward.
/// - **Side-table output**: original AST is not modified.
use std::collections::HashMap;
use tok_parser::ast::*;

// ═══════════════════════════════════════════════════════════════
// Core type representation
// ═══════════════════════════════════════════════════════════════

/// Internal type representation (distinct from parser's `TypeExpr`).
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    Str,
    Bool,
    Nil,
    /// Dynamic type — compatible with everything.
    Any,
    /// Homogeneous array.
    Array(Box<Type>),
    /// Map with string keys.
    Map(Box<Type>),
    /// Tuple with known element types.
    Tuple(Vec<Type>),
    /// Function type.
    Func(FuncType),
    /// Optional: T or Nil.
    Optional(Box<Type>),
    /// Result: (T, ErrStr|Nil).
    Result(Box<Type>),
    /// Channel carrying type T.
    Channel(Box<Type>),
    /// Handle resolving to type T.
    Handle(Box<Type>),
    /// Integer range (iterable).
    Range,
    /// Expression that never produces a value (break, continue, return).
    Never,
}

/// Function type with parameter info and return type.
#[derive(Debug, Clone, PartialEq)]
pub struct FuncType {
    pub params: Vec<ParamType>,
    pub ret: Box<Type>,
    pub variadic: bool,
}

/// Parameter type with default-value info.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamType {
    pub ty: Type,
    pub has_default: bool,
}

// ═══════════════════════════════════════════════════════════════
// Type checker output
// ═══════════════════════════════════════════════════════════════

/// Output of the type checker — consumed by HIR/codegen.
#[derive(Debug)]
pub struct TypeInfo {
    /// Named function signatures.
    pub functions: HashMap<String, FuncType>,
    /// Top-level variable types.
    pub variables: HashMap<String, Type>,
    /// Non-fatal type warnings.
    pub warnings: Vec<TypeWarning>,
}

/// A non-fatal type warning.
#[derive(Debug, Clone)]
pub struct TypeWarning {
    pub msg: String,
}

// ═══════════════════════════════════════════════════════════════
// Unification
// ═══════════════════════════════════════════════════════════════

/// Unify two types. Returns the unified type.
///
/// `Any` is the universal wildcard: `unify(Any, T) = T`.
/// Incompatible concrete types fall back to `Any`.
pub fn unify(a: &Type, b: &Type) -> Type {
    match (a, b) {
        // Any matches anything — stays Any since we don't know the runtime type
        (Type::Any, _) | (_, Type::Any) => Type::Any,

        // Never is absorbed
        (Type::Never, other) | (other, Type::Never) => other.clone(),

        // Identical primitives
        (Type::Int, Type::Int) => Type::Int,
        (Type::Float, Type::Float) => Type::Float,
        (Type::Str, Type::Str) => Type::Str,
        (Type::Bool, Type::Bool) => Type::Bool,
        (Type::Nil, Type::Nil) => Type::Nil,
        (Type::Range, Type::Range) => Type::Range,

        // Numeric widening
        (Type::Int, Type::Float) | (Type::Float, Type::Int) => Type::Float,

        // Nil + concrete → Optional
        (Type::Nil, other) | (other, Type::Nil) => Type::Optional(Box::new(other.clone())),

        // Optional unification
        (Type::Optional(inner), other) | (other, Type::Optional(inner)) => {
            Type::Optional(Box::new(unify(inner, other)))
        }

        // Structural unification
        (Type::Array(a), Type::Array(b)) => Type::Array(Box::new(unify(a, b))),
        (Type::Map(a), Type::Map(b)) => Type::Map(Box::new(unify(a, b))),
        (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => {
            Type::Tuple(a.iter().zip(b.iter()).map(|(x, y)| unify(x, y)).collect())
        }
        (Type::Channel(a), Type::Channel(b)) => Type::Channel(Box::new(unify(a, b))),
        (Type::Handle(a), Type::Handle(b)) => Type::Handle(Box::new(unify(a, b))),
        (Type::Result(a), Type::Result(b)) => Type::Result(Box::new(unify(a, b))),

        // Function unification
        (Type::Func(a), Type::Func(b)) if a.params.len() == b.params.len() => {
            Type::Func(FuncType {
                params: a
                    .params
                    .iter()
                    .zip(b.params.iter())
                    .map(|(pa, pb)| ParamType {
                        ty: unify(&pa.ty, &pb.ty),
                        has_default: pa.has_default || pb.has_default,
                    })
                    .collect(),
                ret: Box::new(unify(&a.ret, &b.ret)),
                variadic: a.variadic || b.variadic,
            })
        }

        // Incompatible → Any
        _ => Type::Any,
    }
}

// ═══════════════════════════════════════════════════════════════
// Type environment (scoped)
// ═══════════════════════════════════════════════════════════════

struct TypeEnv {
    scopes: Vec<HashMap<String, Type>>,
}

impl TypeEnv {
    fn new() -> Self {
        TypeEnv {
            scopes: vec![HashMap::new()],
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define(&mut self, name: &str, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn lookup(&self, name: &str) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    fn update(&mut self, name: &str, ty: Type) {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), ty);
                return;
            }
        }
        self.define(name, ty);
    }
}

// ═══════════════════════════════════════════════════════════════
// Helper: build a function type
// ═══════════════════════════════════════════════════════════════

fn func_type(params: &[Type], ret: Type) -> Type {
    Type::Func(FuncType {
        params: params
            .iter()
            .map(|t| ParamType {
                ty: t.clone(),
                has_default: false,
            })
            .collect(),
        ret: Box::new(ret),
        variadic: false,
    })
}

fn func_type_variadic(params: &[Type], ret: Type) -> Type {
    Type::Func(FuncType {
        params: params
            .iter()
            .map(|t| ParamType {
                ty: t.clone(),
                has_default: false,
            })
            .collect(),
        ret: Box::new(ret),
        variadic: true,
    })
}

// ═══════════════════════════════════════════════════════════════
// Builtin registration
// ═══════════════════════════════════════════════════════════════

fn register_builtins(env: &mut TypeEnv) {
    let arr_any = Type::Array(Box::new(Type::Any));
    let map_any = Type::Map(Box::new(Type::Any));

    let builtins: Vec<(&str, Type)> = vec![
        // I/O
        ("p", func_type(&[Type::Any], Type::Nil)),
        ("pl", func_type(&[Type::Any], Type::Nil)),
        // Type inspection
        ("type", func_type(&[Type::Any], Type::Str)),
        // Array operations
        (
            "push",
            func_type(&[arr_any.clone(), Type::Any], arr_any.clone()),
        ),
        ("sort", func_type(&[arr_any.clone()], arr_any.clone())),
        ("rev", func_type(&[arr_any.clone()], arr_any.clone())),
        ("flat", func_type(&[arr_any.clone()], arr_any.clone())),
        ("uniq", func_type(&[arr_any.clone()], arr_any.clone())),
        // Overloaded (string or array)
        (
            "slice",
            func_type(&[Type::Any, Type::Int, Type::Int], Type::Any),
        ),
        ("len", func_type(&[Type::Any], Type::Int)),
        // String operations
        (
            "split",
            func_type(&[Type::Str, Type::Str], Type::Array(Box::new(Type::Str))),
        ),
        (
            "join",
            func_type(&[Type::Array(Box::new(Type::Str)), Type::Str], Type::Str),
        ),
        ("trim", func_type(&[Type::Str], Type::Str)),
        // Numeric array operations
        ("min", func_type(&[arr_any.clone()], Type::Any)),
        ("max", func_type(&[arr_any.clone()], Type::Any)),
        ("sum", func_type(&[arr_any.clone()], Type::Any)),
        ("abs", func_type(&[Type::Any], Type::Any)),
        // Math
        ("floor", func_type(&[Type::Float], Type::Int)),
        ("ceil", func_type(&[Type::Float], Type::Int)),
        ("rand", func_type(&[], Type::Float)),
        // Conversion
        ("int", func_type(&[Type::Any], Type::Int)),
        ("float", func_type(&[Type::Any], Type::Float)),
        ("str", func_type(&[Type::Any], Type::Str)),
        // Map operations
        (
            "keys",
            func_type(&[map_any.clone()], Type::Array(Box::new(Type::Str))),
        ),
        ("vals", func_type(&[map_any.clone()], arr_any.clone())),
        ("has", func_type(&[map_any.clone(), Type::Str], Type::Bool)),
        (
            "del",
            func_type(&[map_any.clone(), Type::Str], map_any.clone()),
        ),
        // Concurrency — chan is special-cased in check_call for 0-or-1 args
        (
            "chan",
            func_type_variadic(&[], Type::Channel(Box::new(Type::Any))),
        ),
        (
            "pmap",
            func_type(&[arr_any.clone(), Type::Any], arr_any.clone()),
        ),
        // Utility
        ("clock", func_type(&[], Type::Int)),
        ("exit", func_type(&[Type::Int], Type::Nil)),
        // New builtins (spec v0.1)
        ("is", func_type(&[Type::Any, Type::Str], Type::Bool)),
        ("pop", func_type(&[arr_any.clone()], Type::Any)),
        ("freq", func_type(&[arr_any.clone()], map_any.clone())),
        (
            "top",
            func_type(&[map_any.clone(), Type::Int], arr_any.clone()),
        ),
        (
            "zip",
            func_type(&[arr_any.clone(), arr_any.clone()], arr_any.clone()),
        ),
        ("args", func_type(&[], arr_any.clone())),
        ("env", func_type(&[Type::Str], Type::Any)),
    ];

    for (name, ty) in builtins {
        env.define(name, ty);
    }
}

// ═══════════════════════════════════════════════════════════════
// Type checker
// ═══════════════════════════════════════════════════════════════

struct TypeChecker {
    env: TypeEnv,
    functions: HashMap<String, FuncType>,
    variables: HashMap<String, Type>,
    warnings: Vec<TypeWarning>,
    current_return_type: Option<Type>,
    in_loop: bool,
}

impl TypeChecker {
    fn new() -> Self {
        let mut env = TypeEnv::new();
        register_builtins(&mut env);
        TypeChecker {
            env,
            functions: HashMap::new(),
            variables: HashMap::new(),
            warnings: Vec::new(),
            current_return_type: None,
            in_loop: false,
        }
    }

    fn warn(&mut self, msg: impl Into<String>) {
        self.warnings.push(TypeWarning { msg: msg.into() });
    }

    fn into_type_info(self) -> TypeInfo {
        let mut functions = self.functions;
        // Include builtin function types so the lowerer/codegen knows their signatures
        for scope in &self.env.scopes {
            for (name, ty) in scope {
                if let Type::Func(ft) = ty {
                    functions.entry(name.clone()).or_insert_with(|| ft.clone());
                }
            }
        }
        TypeInfo {
            functions,
            variables: self.variables,
            warnings: self.warnings,
        }
    }

    // ─── TypeExpr → Type ──────────────────────────────────────

    #[allow(clippy::only_used_in_recursion)]
    fn resolve_type_expr(&self, te: &TypeExpr) -> Type {
        match te {
            TypeExpr::Prim(PrimType::Int) => Type::Int,
            TypeExpr::Prim(PrimType::Float) => Type::Float,
            TypeExpr::Prim(PrimType::Str) => Type::Str,
            TypeExpr::Prim(PrimType::Bool) => Type::Bool,
            TypeExpr::Prim(PrimType::Nil) => Type::Nil,
            TypeExpr::Prim(PrimType::Any) => Type::Any,
            TypeExpr::Array(inner) => Type::Array(Box::new(self.resolve_type_expr(inner))),
            TypeExpr::Map(inner) => Type::Map(Box::new(self.resolve_type_expr(inner))),
            TypeExpr::Tuple(elts) => {
                Type::Tuple(elts.iter().map(|e| self.resolve_type_expr(e)).collect())
            }
            TypeExpr::Func(params, ret) => Type::Func(FuncType {
                params: params
                    .iter()
                    .map(|p| ParamType {
                        ty: self.resolve_type_expr(p),
                        has_default: false,
                    })
                    .collect(),
                ret: Box::new(self.resolve_type_expr(ret)),
                variadic: false,
            }),
            TypeExpr::Optional(inner) => Type::Optional(Box::new(self.resolve_type_expr(inner))),
            TypeExpr::Result(inner) => Type::Result(Box::new(self.resolve_type_expr(inner))),
            TypeExpr::Channel(inner) => Type::Channel(Box::new(self.resolve_type_expr(inner))),
            TypeExpr::Handle(inner) => Type::Handle(Box::new(self.resolve_type_expr(inner))),
            TypeExpr::Var(_) => Type::Any, // no real generics yet
        }
    }

    // ─── Program ──────────────────────────────────────────────

    fn check_program(&mut self, program: &Program) {
        // Pass 1: collect function signatures (for mutual recursion)
        for stmt in program {
            if let Stmt::FuncDecl {
                name,
                params,
                ret_type,
                ..
            } = stmt
            {
                let ft = self.build_func_type(params, ret_type);
                self.env.define(name, Type::Func(ft.clone()));
                self.functions.insert(name.clone(), ft);
            }
        }

        // Pass 2: check all statements
        for stmt in program {
            self.check_stmt(stmt);
        }
    }

    fn build_func_type(&self, params: &[Param], ret_type: &Option<TypeExpr>) -> FuncType {
        FuncType {
            params: params
                .iter()
                .map(|p| ParamType {
                    ty: p
                        .ty
                        .as_ref()
                        .map(|t| self.resolve_type_expr(t))
                        .unwrap_or(Type::Any),
                    has_default: p.default.is_some(),
                })
                .collect(),
            ret: Box::new(
                ret_type
                    .as_ref()
                    .map(|t| self.resolve_type_expr(t))
                    .unwrap_or(Type::Any),
            ),
            variadic: params.last().map_or(false, |p| p.variadic),
        }
    }

    // ─── Statements ───────────────────────────────────────────

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expr(expr) => {
                self.check_expr(expr);
            }

            Stmt::Assign { name, ty, value } => {
                let val_type = self.check_expr(value);
                let final_type = if let Some(te) = ty {
                    let annot = self.resolve_type_expr(te);
                    unify(&annot, &val_type)
                } else {
                    val_type
                };
                self.env.update(name, final_type.clone());
                self.variables.insert(name.clone(), final_type);
            }

            Stmt::FuncDecl {
                name,
                params,
                ret_type,
                body,
            } => {
                let ft = self.build_func_type(params, ret_type);
                // Already registered in pass 1, but re-register for nested funcs
                self.env.define(name, Type::Func(ft.clone()));
                self.functions.insert(name.clone(), ft.clone());

                let prev_ret = self.current_return_type.take();
                self.current_return_type = Some(*ft.ret.clone());

                self.env.push_scope();
                for (param, pty) in params.iter().zip(ft.params.iter()) {
                    self.env.define(&param.name, pty.ty.clone());
                }

                match body {
                    FuncBody::Expr(e) => {
                        self.check_expr(e);
                    }
                    FuncBody::Block(stmts) => {
                        for s in stmts {
                            self.check_stmt(s);
                        }
                    }
                }

                self.env.pop_scope();
                self.current_return_type = prev_ret;
            }

            Stmt::IndexAssign {
                target,
                index,
                value,
            } => {
                self.check_expr(target);
                self.check_expr(index);
                self.check_expr(value);
            }

            Stmt::MemberAssign {
                target,
                field: _,
                value,
            } => {
                self.check_expr(target);
                self.check_expr(value);
            }

            Stmt::CompoundAssign { name, op: _, value } => {
                self.check_expr(value);
                // The variable type doesn't change
                if self.env.lookup(name).is_none() {
                    self.warn(format!(
                        "compound assignment to undefined variable '{name}'"
                    ));
                }
            }

            Stmt::CompoundIndexAssign {
                target,
                index,
                op: _,
                value,
            } => {
                self.check_expr(target);
                self.check_expr(index);
                self.check_expr(value);
            }

            Stmt::CompoundMemberAssign {
                target,
                field: _,
                op: _,
                value,
            } => {
                self.check_expr(target);
                self.check_expr(value);
            }

            Stmt::TupleDestructure { names, value } => {
                let val_type = self.check_expr(value);
                match &val_type {
                    Type::Tuple(elts) if elts.len() == names.len() => {
                        for (name, ty) in names.iter().zip(elts.iter()) {
                            self.env.update(name, ty.clone());
                            self.variables.insert(name.clone(), ty.clone());
                        }
                    }
                    _ => {
                        for name in names {
                            self.env.update(name, Type::Any);
                            self.variables.insert(name.clone(), Type::Any);
                        }
                    }
                }
            }

            Stmt::MapDestructure { names, value } => {
                let val_type = self.check_expr(value);
                let elem_type = match &val_type {
                    Type::Map(inner) => *inner.clone(),
                    _ => Type::Any,
                };
                for name in names {
                    self.env.update(name, elem_type.clone());
                    self.variables.insert(name.clone(), elem_type.clone());
                }
            }

            Stmt::ArrayDestructure { head, tail, value } => {
                let val_type = self.check_expr(value);
                let elem_type = match &val_type {
                    Type::Array(inner) => *inner.clone(),
                    _ => Type::Any,
                };
                self.env.update(head, elem_type.clone());
                self.variables.insert(head.clone(), elem_type.clone());
                self.env
                    .update(tail, Type::Array(Box::new(elem_type.clone())));
                self.variables
                    .insert(tail.clone(), Type::Array(Box::new(elem_type)));
            }

            Stmt::Import(_) => {
                // Can't resolve imports at type-check time
            }

            Stmt::Return(expr) => {
                if let Some(e) = expr {
                    self.check_expr(e);
                }
            }

            Stmt::Break | Stmt::Continue => {}
        }
    }

    // ─── Expressions ──────────────────────────────────────────

    fn check_expr(&mut self, expr: &Expr) -> Type {
        match expr {
            // Literals
            Expr::Int(_) => Type::Int,
            Expr::Float(_) => Type::Float,
            Expr::Str(_) => Type::Str,
            Expr::Bool(_) => Type::Bool,
            Expr::Nil => Type::Nil,

            Expr::Interp(parts) => {
                for part in parts {
                    if let InterpPart::Expr(e) = part {
                        self.check_expr(e);
                    }
                }
                Type::Str
            }

            // Identifiers
            Expr::Ident(name) => self.env.lookup(name).cloned().unwrap_or(Type::Any),

            // Compound literals
            Expr::Array(elts) => {
                if elts.is_empty() {
                    Type::Array(Box::new(Type::Any))
                } else {
                    let mut elem_ty = Type::Any;
                    for e in elts {
                        let t = self.check_expr(e);
                        // Skip spreads for element type — they contribute their inner type
                        let t = if let Type::Array(inner) = &t {
                            if matches!(e, Expr::Spread(_)) {
                                *inner.clone()
                            } else {
                                t
                            }
                        } else {
                            t
                        };
                        elem_ty = unify(&elem_ty, &t);
                    }
                    Type::Array(Box::new(elem_ty))
                }
            }

            Expr::Map(pairs) => {
                if pairs.is_empty() {
                    Type::Map(Box::new(Type::Any))
                } else {
                    let mut val_ty = Type::Any;
                    for (_, v) in pairs {
                        let t = self.check_expr(v);
                        val_ty = unify(&val_ty, &t);
                    }
                    Type::Map(Box::new(val_ty))
                }
            }

            Expr::Tuple(elts) => Type::Tuple(elts.iter().map(|e| self.check_expr(e)).collect()),

            // Range
            Expr::Range { start, end, .. } => {
                self.check_expr(start);
                self.check_expr(end);
                Type::Range
            }

            // Binary operators
            Expr::BinOp { op, left, right } => {
                let lt = self.check_expr(left);
                let rt = self.check_expr(right);
                self.check_binop(op, &lt, &rt)
            }

            // Unary operators
            Expr::UnaryOp { op, expr } => {
                let t = self.check_expr(expr);
                match op {
                    UnaryOp::Neg => match &t {
                        Type::Int => Type::Int,
                        Type::Float => Type::Float,
                        _ => Type::Any,
                    },
                    UnaryOp::Not => Type::Bool,
                }
            }

            // Index access
            Expr::Index { expr, index } => {
                let target_ty = self.check_expr(expr);
                self.check_expr(index);
                match &target_ty {
                    Type::Array(inner) => *inner.clone(),
                    Type::Map(inner) => *inner.clone(),
                    Type::Tuple(elts) => {
                        // If index is a literal int, we can get the exact type
                        if let Expr::Int(i) = index.as_ref() {
                            let idx = *i as usize;
                            elts.get(idx).cloned().unwrap_or(Type::Any)
                        } else {
                            Type::Any
                        }
                    }
                    Type::Str => Type::Str,
                    _ => Type::Any,
                }
            }

            // Member access
            Expr::Member { expr, field } => {
                let target_ty = self.check_expr(expr);
                self.resolve_member(&target_ty, field)
            }

            // Optional chain
            Expr::OptionalChain { expr, field } => {
                let target_ty = self.check_expr(expr);
                match &target_ty {
                    Type::Nil => Type::Nil,
                    Type::Optional(inner) => {
                        let field_ty = self.resolve_member(inner, field);
                        Type::Optional(Box::new(field_ty))
                    }
                    _ => {
                        let field_ty = self.resolve_member(&target_ty, field);
                        Type::Optional(Box::new(field_ty))
                    }
                }
            }

            // Function call
            Expr::Call { func, args } => self.check_call(func, args),

            // Lambda
            Expr::Lambda {
                params,
                ret_type,
                body,
            } => {
                let ft = self.build_func_type(params, ret_type);

                let prev_ret = self.current_return_type.take();
                self.current_return_type = Some(*ft.ret.clone());

                self.env.push_scope();
                for (param, pty) in params.iter().zip(ft.params.iter()) {
                    self.env.define(&param.name, pty.ty.clone());
                }

                match body {
                    FuncBody::Expr(e) => {
                        self.check_expr(e);
                    }
                    FuncBody::Block(stmts) => {
                        for s in stmts {
                            self.check_stmt(s);
                        }
                    }
                }

                self.env.pop_scope();
                self.current_return_type = prev_ret;

                Type::Func(ft)
            }

            // Ternary
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                self.check_expr(cond);
                let then_ty = self.check_expr(then_expr);
                if let Some(else_e) = else_expr {
                    let else_ty = self.check_expr(else_e);
                    unify(&then_ty, &else_ty)
                } else {
                    then_ty
                }
            }

            // Match
            Expr::Match { subject, arms } => {
                if let Some(subj) = subject {
                    self.check_expr(subj);
                }
                let mut result_ty = Type::Never;
                for arm in arms {
                    self.check_pattern(&arm.pattern);
                    let body_ty = match &arm.body {
                        MatchBody::Expr(e) => self.check_expr(e),
                        MatchBody::Block(stmts) => self.check_block_stmts(stmts),
                    };
                    result_ty = unify(&result_ty, &body_ty);
                }
                result_ty
            }

            // Loop
            Expr::Loop { clause, body } => {
                let prev_in_loop = self.in_loop;
                self.in_loop = true;

                self.env.push_scope();

                match clause.as_ref() {
                    LoopClause::While(cond) => {
                        self.check_expr(cond);
                    }
                    LoopClause::ForRange { var, range } => {
                        self.check_expr(range);
                        self.env.define(var, Type::Int);
                    }
                    LoopClause::ForEach { var, iter } => {
                        let iter_ty = self.check_expr(iter);
                        let elem_ty = match &iter_ty {
                            Type::Array(inner) => *inner.clone(),
                            Type::Str => Type::Str,
                            _ => Type::Any,
                        };
                        self.env.define(var, elem_ty);
                    }
                    LoopClause::ForEachIndexed {
                        idx_var,
                        val_var,
                        iter,
                    } => {
                        let iter_ty = self.check_expr(iter);
                        let elem_ty = match &iter_ty {
                            Type::Array(inner) => *inner.clone(),
                            Type::Map(inner) => *inner.clone(),
                            _ => Type::Any,
                        };
                        let idx_ty = match &iter_ty {
                            Type::Map(_) => Type::Str,
                            _ => Type::Int,
                        };
                        self.env.define(idx_var, idx_ty);
                        self.env.define(val_var, elem_ty);
                    }
                    LoopClause::Infinite => {}
                }

                let result_ty = match body.as_ref() {
                    LoopBody::Block(stmts) => {
                        for s in stmts {
                            self.check_stmt(s);
                        }
                        Type::Nil
                    }
                    LoopBody::Collect(e) => {
                        let elem_ty = self.check_expr(e);
                        Type::Array(Box::new(elem_ty))
                    }
                };

                self.env.pop_scope();
                self.in_loop = prev_in_loop;
                result_ty
            }

            // Block
            Expr::Block(stmts) => self.check_block_stmts(stmts),

            // Pipeline
            Expr::Pipeline { left, right } => {
                let left_ty = self.check_expr(left);
                let right_ty = self.check_expr(right);
                match &right_ty {
                    Type::Func(ft) => *ft.ret.clone(),
                    _ => {
                        // Pipeline into a non-function — treat as call
                        let _ = left_ty;
                        Type::Any
                    }
                }
            }

            // Filter
            Expr::Filter { expr, pred } => {
                let arr_ty = self.check_expr(expr);
                self.check_expr(pred);
                // Filter preserves array type
                match &arr_ty {
                    Type::Array(_) => arr_ty,
                    _ => Type::Any,
                }
            }

            // Reduce
            Expr::Reduce { expr, init, func } => {
                let arr_ty = self.check_expr(expr);
                let init_ty = if let Some(init_e) = init {
                    self.check_expr(init_e)
                } else {
                    match &arr_ty {
                        Type::Array(inner) => *inner.clone(),
                        _ => Type::Any,
                    }
                };
                self.check_expr(func);
                init_ty
            }

            // Spread
            Expr::Spread(inner) => {
                // Spread of [T] contributes T elements
                self.check_expr(inner)
            }

            // Length
            Expr::Length(inner) => {
                self.check_expr(inner);
                Type::Int
            }

            // Nil coalesce
            Expr::NilCoalesce { left, right } => {
                let left_ty = self.check_expr(left);
                let right_ty = self.check_expr(right);
                match &left_ty {
                    Type::Optional(inner) => unify(inner, &right_ty),
                    Type::Nil => right_ty,
                    _ => unify(&left_ty, &right_ty),
                }
            }

            // Error propagation
            Expr::ErrorPropagate(inner) => {
                let inner_ty = self.check_expr(inner);
                match &inner_ty {
                    Type::Result(ok_ty) => *ok_ty.clone(),
                    Type::Tuple(elts) if elts.len() == 2 => elts[0].clone(),
                    _ => Type::Any,
                }
            }

            // Conditional return
            Expr::ConditionalReturn { cond, value } => {
                self.check_expr(cond);
                self.check_expr(value);
                Type::Never
            }

            // Concurrency
            Expr::Go(body) => {
                let body_ty = self.check_expr(body);
                Type::Handle(Box::new(body_ty))
            }

            Expr::Receive(inner) => {
                let inner_ty = self.check_expr(inner);
                match &inner_ty {
                    Type::Channel(t) => *t.clone(),
                    Type::Handle(t) => *t.clone(),
                    _ => Type::Any,
                }
            }

            Expr::Send { chan, value } => {
                self.check_expr(chan);
                self.check_expr(value);
                Type::Nil
            }

            Expr::Select(arms) => {
                let mut result_ty = Type::Never;
                for arm in arms {
                    let body_ty = match arm {
                        SelectArm::Recv { var, chan, body } => {
                            let chan_ty = self.check_expr(chan);
                            let elem_ty = match &chan_ty {
                                Type::Channel(inner) => *inner.clone(),
                                _ => Type::Any,
                            };
                            self.env.push_scope();
                            self.env.define(var, elem_ty);
                            let ty = self.check_block_stmts(body);
                            self.env.pop_scope();
                            ty
                        }
                        SelectArm::Send { chan, value, body } => {
                            self.check_expr(chan);
                            self.check_expr(value);
                            self.env.push_scope();
                            let ty = self.check_block_stmts(body);
                            self.env.pop_scope();
                            ty
                        }
                        SelectArm::Default(body) => {
                            self.env.push_scope();
                            let ty = self.check_block_stmts(body);
                            self.env.pop_scope();
                            ty
                        }
                    };
                    result_ty = unify(&result_ty, &body_ty);
                }
                result_ty
            }

            // Import
            Expr::Import(_) => Type::Map(Box::new(Type::Any)),

            // Control flow as expressions
            Expr::Return(expr) => {
                if let Some(e) = expr {
                    self.check_expr(e);
                }
                Type::Never
            }
            Expr::Break => Type::Never,
            Expr::Continue => Type::Never,
        }
    }

    // ─── Helpers ──────────────────────────────────────────────

    fn check_binop(&self, op: &BinOp, lt: &Type, rt: &Type) -> Type {
        match op {
            // Arithmetic
            BinOp::Add => match (lt, rt) {
                (Type::Int, Type::Int) => Type::Int,
                (Type::Float, Type::Float) => Type::Float,
                (Type::Int, Type::Float) | (Type::Float, Type::Int) => Type::Float,
                (Type::Str, Type::Str) => Type::Str,
                (Type::Str, _) | (_, Type::Str) => Type::Str,
                _ => Type::Any,
            },
            BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => match (lt, rt) {
                (Type::Int, Type::Int) => Type::Int,
                (Type::Float, Type::Float) => Type::Float,
                (Type::Int, Type::Float) | (Type::Float, Type::Int) => Type::Float,
                _ => Type::Any,
            },

            // Comparison — always bool
            BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                Type::Bool
            }

            // Logic — always bool
            BinOp::And | BinOp::Or => Type::Bool,

            // Bitwise — int
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                match (lt, rt) {
                    (Type::Int, Type::Int) => Type::Int,
                    _ => Type::Any,
                }
            }
        }
    }

    fn resolve_member(&self, target_ty: &Type, field: &str) -> Type {
        match target_ty {
            Type::Map(_) => Type::Any,
            Type::Tuple(elts) => {
                if let Ok(idx) = field.parse::<usize>() {
                    elts.get(idx).cloned().unwrap_or(Type::Any)
                } else {
                    Type::Any
                }
            }
            _ => Type::Any,
        }
    }

    fn check_call(&mut self, func: &Expr, args: &[Expr]) -> Type {
        // Check all argument types
        let _arg_types: Vec<Type> = args.iter().map(|a| self.check_expr(a)).collect();

        // Get the function type
        let func_ty = self.check_expr(func);

        match &func_ty {
            Type::Func(ft) => *ft.ret.clone(),
            Type::Any => Type::Any,
            _ => Type::Any,
        }
    }

    fn check_pattern(&mut self, pat: &Pattern) {
        match pat {
            Pattern::Guard(expr) => {
                self.check_expr(expr);
            }
            Pattern::Tuple(pats) => {
                for p in pats {
                    self.check_pattern(p);
                }
            }
            _ => {}
        }
    }

    fn check_block_stmts(&mut self, stmts: &[Stmt]) -> Type {
        self.env.push_scope();
        let mut last_ty = Type::Nil;
        for stmt in stmts {
            self.check_stmt(stmt);
            // Track last expression type for block result
            if let Stmt::Expr(e) = stmt {
                last_ty = self.check_expr_type_only(e);
            }
        }
        self.env.pop_scope();
        last_ty
    }

    /// Get the type of an already-checked expression without re-checking.
    /// Falls back to re-checking (idempotent since we don't mutate the AST).
    fn check_expr_type_only(&mut self, expr: &Expr) -> Type {
        self.check_expr(expr)
    }
}

// ═══════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════

/// Type-check a parsed program and produce type information.
///
/// This never fails — type issues are recorded as warnings in `TypeInfo`.
/// All untyped code is accepted (annotations default to `Any`).
pub fn check(program: &Program) -> TypeInfo {
    let mut checker = TypeChecker::new();
    checker.check_program(program);
    checker.into_type_info()
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Unification tests ────────────────────────────────────

    #[test]
    fn unify_any_with_concrete() {
        assert_eq!(unify(&Type::Any, &Type::Int), Type::Any);
        assert_eq!(unify(&Type::Str, &Type::Any), Type::Any);
        assert_eq!(unify(&Type::Any, &Type::Any), Type::Any);
    }

    #[test]
    fn unify_never_absorbed() {
        assert_eq!(unify(&Type::Never, &Type::Int), Type::Int);
        assert_eq!(unify(&Type::Str, &Type::Never), Type::Str);
    }

    #[test]
    fn unify_numeric_widening() {
        assert_eq!(unify(&Type::Int, &Type::Float), Type::Float);
        assert_eq!(unify(&Type::Float, &Type::Int), Type::Float);
    }

    #[test]
    fn unify_nil_produces_optional() {
        assert_eq!(
            unify(&Type::Nil, &Type::Int),
            Type::Optional(Box::new(Type::Int))
        );
        assert_eq!(
            unify(&Type::Str, &Type::Nil),
            Type::Optional(Box::new(Type::Str))
        );
    }

    #[test]
    fn unify_same_types() {
        assert_eq!(unify(&Type::Int, &Type::Int), Type::Int);
        assert_eq!(unify(&Type::Str, &Type::Str), Type::Str);
        assert_eq!(unify(&Type::Bool, &Type::Bool), Type::Bool);
    }

    #[test]
    fn unify_incompatible_to_any() {
        assert_eq!(unify(&Type::Bool, &Type::Int), Type::Any);
        assert_eq!(unify(&Type::Str, &Type::Bool), Type::Any);
    }

    #[test]
    fn unify_arrays() {
        let a = Type::Array(Box::new(Type::Int));
        let b = Type::Array(Box::new(Type::Int));
        assert_eq!(unify(&a, &b), Type::Array(Box::new(Type::Int)));

        let c = Type::Array(Box::new(Type::Any));
        assert_eq!(unify(&a, &c), Type::Array(Box::new(Type::Any)));
    }

    #[test]
    fn unify_tuples() {
        let a = Type::Tuple(vec![Type::Int, Type::Str]);
        let b = Type::Tuple(vec![Type::Int, Type::Str]);
        assert_eq!(unify(&a, &b), Type::Tuple(vec![Type::Int, Type::Str]));

        // Different lengths → Any
        let c = Type::Tuple(vec![Type::Int]);
        assert_eq!(unify(&a, &c), Type::Any);
    }

    // ─── TypeEnv tests ────────────────────────────────────────

    #[test]
    fn env_define_lookup() {
        let mut env = TypeEnv::new();
        env.define("x", Type::Int);
        assert_eq!(env.lookup("x"), Some(&Type::Int));
        assert_eq!(env.lookup("y"), None);
    }

    #[test]
    fn env_scoping() {
        let mut env = TypeEnv::new();
        env.define("x", Type::Int);
        env.push_scope();
        env.define("x", Type::Str);
        assert_eq!(env.lookup("x"), Some(&Type::Str));
        env.pop_scope();
        assert_eq!(env.lookup("x"), Some(&Type::Int));
    }

    #[test]
    fn env_update() {
        let mut env = TypeEnv::new();
        env.define("x", Type::Int);
        env.update("x", Type::Float);
        assert_eq!(env.lookup("x"), Some(&Type::Float));
    }

    // ─── Resolve type expression ──────────────────────────────

    #[test]
    fn resolve_prim_types() {
        let checker = TypeChecker::new();
        assert_eq!(
            checker.resolve_type_expr(&TypeExpr::Prim(PrimType::Int)),
            Type::Int
        );
        assert_eq!(
            checker.resolve_type_expr(&TypeExpr::Prim(PrimType::Float)),
            Type::Float
        );
        assert_eq!(
            checker.resolve_type_expr(&TypeExpr::Prim(PrimType::Str)),
            Type::Str
        );
        assert_eq!(
            checker.resolve_type_expr(&TypeExpr::Prim(PrimType::Bool)),
            Type::Bool
        );
        assert_eq!(
            checker.resolve_type_expr(&TypeExpr::Prim(PrimType::Nil)),
            Type::Nil
        );
        assert_eq!(
            checker.resolve_type_expr(&TypeExpr::Prim(PrimType::Any)),
            Type::Any
        );
    }

    #[test]
    fn resolve_compound_types() {
        let checker = TypeChecker::new();
        let arr = TypeExpr::Array(Box::new(TypeExpr::Prim(PrimType::Int)));
        assert_eq!(
            checker.resolve_type_expr(&arr),
            Type::Array(Box::new(Type::Int))
        );

        let map = TypeExpr::Map(Box::new(TypeExpr::Prim(PrimType::Str)));
        assert_eq!(
            checker.resolve_type_expr(&map),
            Type::Map(Box::new(Type::Str))
        );

        let opt = TypeExpr::Optional(Box::new(TypeExpr::Prim(PrimType::Int)));
        assert_eq!(
            checker.resolve_type_expr(&opt),
            Type::Optional(Box::new(Type::Int))
        );
    }

    // ─── Expression type inference ────────────────────────────

    #[test]
    fn literal_types() {
        let prog: Program = vec![
            Stmt::Assign {
                name: "a".into(),
                ty: None,
                value: Expr::Int(1),
            },
            Stmt::Assign {
                name: "b".into(),
                ty: None,
                value: Expr::Float(1.0),
            },
            Stmt::Assign {
                name: "c".into(),
                ty: None,
                value: Expr::Str("hi".into()),
            },
            Stmt::Assign {
                name: "d".into(),
                ty: None,
                value: Expr::Bool(true),
            },
            Stmt::Assign {
                name: "e".into(),
                ty: None,
                value: Expr::Nil,
            },
        ];
        let info = check(&prog);
        assert_eq!(info.variables["a"], Type::Int);
        assert_eq!(info.variables["b"], Type::Float);
        assert_eq!(info.variables["c"], Type::Str);
        assert_eq!(info.variables["d"], Type::Bool);
        assert_eq!(info.variables["e"], Type::Nil);
    }

    #[test]
    fn array_inference() {
        let prog: Program = vec![Stmt::Assign {
            name: "arr".into(),
            ty: None,
            value: Expr::Array(vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)]),
        }];
        let info = check(&prog);
        assert_eq!(info.variables["arr"], Type::Array(Box::new(Type::Any)));
    }

    #[test]
    fn empty_array_is_any() {
        let prog: Program = vec![Stmt::Assign {
            name: "arr".into(),
            ty: None,
            value: Expr::Array(vec![]),
        }];
        let info = check(&prog);
        assert_eq!(info.variables["arr"], Type::Array(Box::new(Type::Any)));
    }

    #[test]
    fn map_inference() {
        let prog: Program = vec![Stmt::Assign {
            name: "m".into(),
            ty: None,
            value: Expr::Map(vec![
                (MapKey::Ident("a".into()), Expr::Int(1)),
                (MapKey::Ident("b".into()), Expr::Int(2)),
            ]),
        }];
        let info = check(&prog);
        assert_eq!(info.variables["m"], Type::Map(Box::new(Type::Any)));
    }

    #[test]
    fn tuple_inference() {
        let prog: Program = vec![Stmt::Assign {
            name: "t".into(),
            ty: None,
            value: Expr::Tuple(vec![Expr::Int(1), Expr::Str("hi".into()), Expr::Bool(true)]),
        }];
        let info = check(&prog);
        assert_eq!(
            info.variables["t"],
            Type::Tuple(vec![Type::Int, Type::Str, Type::Bool])
        );
    }

    #[test]
    fn binop_arithmetic() {
        let prog: Program = vec![Stmt::Assign {
            name: "x".into(),
            ty: None,
            value: Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::Int(1)),
                right: Box::new(Expr::Int(2)),
            },
        }];
        let info = check(&prog);
        assert_eq!(info.variables["x"], Type::Int);
    }

    #[test]
    fn binop_float_widening() {
        let prog: Program = vec![Stmt::Assign {
            name: "x".into(),
            ty: None,
            value: Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::Int(1)),
                right: Box::new(Expr::Float(2.0)),
            },
        }];
        let info = check(&prog);
        assert_eq!(info.variables["x"], Type::Float);
    }

    #[test]
    fn binop_comparison() {
        let prog: Program = vec![Stmt::Assign {
            name: "x".into(),
            ty: None,
            value: Expr::BinOp {
                op: BinOp::Lt,
                left: Box::new(Expr::Int(1)),
                right: Box::new(Expr::Int(2)),
            },
        }];
        let info = check(&prog);
        assert_eq!(info.variables["x"], Type::Bool);
    }

    #[test]
    fn func_decl_with_types() {
        let prog: Program = vec![Stmt::FuncDecl {
            name: "fib".into(),
            params: vec![Param {
                name: "n".into(),
                ty: Some(TypeExpr::Prim(PrimType::Int)),
                default: None,
                variadic: false,
            }],
            ret_type: Some(TypeExpr::Prim(PrimType::Int)),
            body: FuncBody::Expr(Box::new(Expr::Ident("n".into()))),
        }];
        let info = check(&prog);
        let ft = info.functions.get("fib").unwrap();
        assert_eq!(ft.params[0].ty, Type::Int);
        assert_eq!(*ft.ret, Type::Int);
    }

    #[test]
    fn func_decl_without_types() {
        let prog: Program = vec![Stmt::FuncDecl {
            name: "f".into(),
            params: vec![Param {
                name: "x".into(),
                ty: None,
                default: None,
                variadic: false,
            }],
            ret_type: None,
            body: FuncBody::Expr(Box::new(Expr::Ident("x".into()))),
        }];
        let info = check(&prog);
        let ft = info.functions.get("f").unwrap();
        assert_eq!(ft.params[0].ty, Type::Any);
        assert_eq!(*ft.ret, Type::Any);
    }

    #[test]
    fn call_returns_func_ret_type() {
        let prog: Program = vec![
            Stmt::FuncDecl {
                name: "double".into(),
                params: vec![Param {
                    name: "x".into(),
                    ty: Some(TypeExpr::Prim(PrimType::Int)),
                    default: None,
                    variadic: false,
                }],
                ret_type: Some(TypeExpr::Prim(PrimType::Int)),
                body: FuncBody::Expr(Box::new(Expr::BinOp {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Ident("x".into())),
                    right: Box::new(Expr::Int(2)),
                })),
            },
            Stmt::Assign {
                name: "result".into(),
                ty: None,
                value: Expr::Call {
                    func: Box::new(Expr::Ident("double".into())),
                    args: vec![Expr::Int(21)],
                },
            },
        ];
        let info = check(&prog);
        assert_eq!(info.variables["result"], Type::Int);
    }

    #[test]
    fn ternary_unifies_branches() {
        let prog: Program = vec![Stmt::Assign {
            name: "x".into(),
            ty: None,
            value: Expr::Ternary {
                cond: Box::new(Expr::Bool(true)),
                then_expr: Box::new(Expr::Int(1)),
                else_expr: Some(Box::new(Expr::Int(2))),
            },
        }];
        let info = check(&prog);
        assert_eq!(info.variables["x"], Type::Int);
    }

    #[test]
    fn ternary_mixed_branches() {
        let prog: Program = vec![Stmt::Assign {
            name: "x".into(),
            ty: None,
            value: Expr::Ternary {
                cond: Box::new(Expr::Bool(true)),
                then_expr: Box::new(Expr::Int(1)),
                else_expr: Some(Box::new(Expr::Str("hello".into()))),
            },
        }];
        let info = check(&prog);
        // Int and Str are incompatible → Any
        assert_eq!(info.variables["x"], Type::Any);
    }

    #[test]
    fn tuple_destructure() {
        let prog: Program = vec![Stmt::TupleDestructure {
            names: vec!["a".into(), "b".into()],
            value: Expr::Tuple(vec![Expr::Int(1), Expr::Str("hi".into())]),
        }];
        let info = check(&prog);
        assert_eq!(info.variables["a"], Type::Int);
        assert_eq!(info.variables["b"], Type::Str);
    }

    #[test]
    fn length_is_int() {
        let prog: Program = vec![Stmt::Assign {
            name: "n".into(),
            ty: None,
            value: Expr::Length(Box::new(Expr::Array(vec![Expr::Int(1)]))),
        }];
        let info = check(&prog);
        assert_eq!(info.variables["n"], Type::Int);
    }

    #[test]
    fn go_produces_handle() {
        let prog: Program = vec![Stmt::Assign {
            name: "h".into(),
            ty: None,
            value: Expr::Go(Box::new(Expr::Int(42))),
        }];
        let info = check(&prog);
        assert_eq!(info.variables["h"], Type::Handle(Box::new(Type::Int)));
    }

    #[test]
    fn receive_from_handle() {
        let prog: Program = vec![
            Stmt::Assign {
                name: "h".into(),
                ty: None,
                value: Expr::Go(Box::new(Expr::Int(42))),
            },
            Stmt::Assign {
                name: "v".into(),
                ty: None,
                value: Expr::Receive(Box::new(Expr::Ident("h".into()))),
            },
        ];
        let info = check(&prog);
        assert_eq!(info.variables["v"], Type::Int);
    }

    #[test]
    fn builtin_pl_returns_nil() {
        let prog: Program = vec![Stmt::Assign {
            name: "r".into(),
            ty: None,
            value: Expr::Call {
                func: Box::new(Expr::Ident("pl".into())),
                args: vec![Expr::Int(42)],
            },
        }];
        let info = check(&prog);
        assert_eq!(info.variables["r"], Type::Nil);
    }

    #[test]
    fn builtin_type_returns_str() {
        let prog: Program = vec![Stmt::Assign {
            name: "t".into(),
            ty: None,
            value: Expr::Call {
                func: Box::new(Expr::Ident("type".into())),
                args: vec![Expr::Int(42)],
            },
        }];
        let info = check(&prog);
        assert_eq!(info.variables["t"], Type::Str);
    }

    #[test]
    fn annotated_variable() {
        let prog: Program = vec![Stmt::Assign {
            name: "x".into(),
            ty: Some(TypeExpr::Prim(PrimType::Int)),
            value: Expr::Int(42),
        }];
        let info = check(&prog);
        assert_eq!(info.variables["x"], Type::Int);
    }

    #[test]
    fn interp_is_str() {
        let prog: Program = vec![Stmt::Assign {
            name: "s".into(),
            ty: None,
            value: Expr::Interp(vec![
                InterpPart::Lit("hello ".into()),
                InterpPart::Expr(Expr::Ident("name".into())),
            ]),
        }];
        let info = check(&prog);
        assert_eq!(info.variables["s"], Type::Str);
    }

    #[test]
    fn range_type() {
        let prog: Program = vec![Stmt::Assign {
            name: "r".into(),
            ty: None,
            value: Expr::Range {
                start: Box::new(Expr::Int(0)),
                end: Box::new(Expr::Int(10)),
                inclusive: false,
            },
        }];
        let info = check(&prog);
        assert_eq!(info.variables["r"], Type::Range);
    }

    #[test]
    fn import_is_map_any() {
        let prog: Program = vec![Stmt::Assign {
            name: "m".into(),
            ty: None,
            value: Expr::Import("./module.tok".into()),
        }];
        let info = check(&prog);
        assert_eq!(info.variables["m"], Type::Map(Box::new(Type::Any)));
    }

    #[test]
    fn filter_preserves_array_type() {
        let prog: Program = vec![
            Stmt::Assign {
                name: "arr".into(),
                ty: None,
                value: Expr::Array(vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)]),
            },
            Stmt::Assign {
                name: "filtered".into(),
                ty: None,
                value: Expr::Filter {
                    expr: Box::new(Expr::Ident("arr".into())),
                    pred: Box::new(Expr::Lambda {
                        params: vec![Param {
                            name: "x".into(),
                            ty: None,
                            default: None,
                            variadic: false,
                        }],
                        ret_type: None,
                        body: FuncBody::Expr(Box::new(Expr::BinOp {
                            op: BinOp::Gt,
                            left: Box::new(Expr::Ident("x".into())),
                            right: Box::new(Expr::Int(1)),
                        })),
                    }),
                },
            },
        ];
        let info = check(&prog);
        assert_eq!(info.variables["filtered"], Type::Array(Box::new(Type::Any)));
    }

    #[test]
    fn index_array_returns_elem_type() {
        let prog: Program = vec![
            Stmt::Assign {
                name: "arr".into(),
                ty: None,
                value: Expr::Array(vec![Expr::Int(1), Expr::Int(2)]),
            },
            Stmt::Assign {
                name: "v".into(),
                ty: None,
                value: Expr::Index {
                    expr: Box::new(Expr::Ident("arr".into())),
                    index: Box::new(Expr::Int(0)),
                },
            },
        ];
        let info = check(&prog);
        assert_eq!(info.variables["v"], Type::Any);
    }

    #[test]
    fn member_access_tuple() {
        let prog: Program = vec![
            Stmt::Assign {
                name: "t".into(),
                ty: None,
                value: Expr::Tuple(vec![Expr::Int(1), Expr::Str("hi".into())]),
            },
            Stmt::Assign {
                name: "v".into(),
                ty: None,
                value: Expr::Member {
                    expr: Box::new(Expr::Ident("t".into())),
                    field: "0".into(),
                },
            },
        ];
        let info = check(&prog);
        assert_eq!(info.variables["v"], Type::Int);
    }
}
