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

// ═══════════════════════════════════════════════════════════════
// Lowerer state
// ═══════════════════════════════════════════════════════════════

struct Lowerer<'a> {
    type_info: &'a TypeInfo,
    tmp_counter: u32,
    /// Local variable type scopes (for function params and local vars).
    scopes: Vec<std::collections::HashMap<String, Type>>,
}

impl<'a> Lowerer<'a> {
    fn new(type_info: &'a TypeInfo) -> Self {
        Lowerer {
            type_info,
            tmp_counter: 0,
            scopes: Vec::new(),
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(std::collections::HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_local(&mut self, name: &str, ty: Type) {
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
    fn gensym(&mut self) -> String {
        self.tmp_counter += 1;
        format!("_tmp{}", self.tmp_counter)
    }

    // ═══════════════════════════════════════════════════════════
    // Type helpers
    // ═══════════════════════════════════════════════════════════

    /// Look up the type of a variable from local scopes, TypeInfo variables, or functions.
    fn var_type(&self, name: &str) -> Type {
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
    fn func_ret_type(&self, name: &str) -> Type {
        self.type_info
            .functions
            .get(name)
            .map(|ft| *ft.ret.clone())
            .unwrap_or(Type::Any)
    }

    /// Infer the type of an AST expression (simplified, best-effort).
    fn infer_expr_type(&self, expr: &Expr) -> Type {
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
            Expr::Import(_) => Type::Map(Box::new(Type::Any)),
            Expr::Return(_) | Expr::Break | Expr::Continue => Type::Never,
        }
    }

    fn infer_binop_type(&self, op: &BinOp, lt: &Type, rt: &Type) -> Type {
        tok_types::infer_binop_type(op, lt, rt)
    }

    fn infer_member_type(&self, target_ty: &Type, field: &str) -> Type {
        tok_types::infer_member_type(target_ty, field)
    }

    // ═══════════════════════════════════════════════════════════
    // BinOp conversion
    // ═══════════════════════════════════════════════════════════

    fn lower_binop(&self, op: &BinOp) -> HirBinOp {
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

    fn lower_unaryop(&self, op: &UnaryOp) -> HirUnaryOp {
        match op {
            UnaryOp::Neg => HirUnaryOp::Neg,
            UnaryOp::Not => HirUnaryOp::Not,
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Statement lowering
    // ═══════════════════════════════════════════════════════════

    fn lower_program(&mut self, program: &Program) -> HirProgram {
        let mut out = Vec::new();
        for stmt in program {
            self.lower_stmt(stmt, &mut out);
        }
        out
    }

    fn lower_stmt(&mut self, stmt: &Stmt, out: &mut Vec<HirStmt>) {
        match stmt {
            Stmt::Expr(expr) => {
                let hir_expr = self.lower_expr(expr);
                out.push(HirStmt::Expr(hir_expr));
            }

            Stmt::Assign { name, value, .. } => {
                let hir_value = self.lower_expr(value);
                let ty = hir_value.ty.clone();
                // Track the local variable type for subsequent references
                self.define_local(name, ty.clone());
                out.push(HirStmt::Assign {
                    name: name.clone(),
                    ty,
                    value: hir_value,
                });
            }

            Stmt::FuncDecl {
                name,
                params,
                ret_type: _,
                body,
            } => {
                let hir_params = self.lower_params(params);
                let ret_type = self.func_ret_type(name);
                // Push scope for function body with param types
                self.push_scope();
                for hp in &hir_params {
                    self.define_local(&hp.name, hp.ty.clone());
                }
                let hir_body = self.lower_func_body(body);
                self.pop_scope();
                out.push(HirStmt::FuncDecl {
                    name: name.clone(),
                    params: hir_params,
                    ret_type,
                    body: hir_body,
                });
            }

            Stmt::IndexAssign {
                target,
                index,
                value,
            } => {
                let hir_target = self.lower_expr(target);
                let hir_index = self.lower_expr(index);
                let hir_value = self.lower_expr(value);
                out.push(HirStmt::IndexAssign {
                    target: hir_target,
                    index: hir_index,
                    value: hir_value,
                });
            }

            Stmt::MemberAssign {
                target,
                field,
                value,
            } => {
                let hir_target = self.lower_expr(target);
                let hir_value = self.lower_expr(value);
                out.push(HirStmt::MemberAssign {
                    target: hir_target,
                    field: field.clone(),
                    value: hir_value,
                });
            }

            // Desugar: x += 1 -> x = x + 1  (or x <<= v -> x = push(x, v))
            Stmt::CompoundAssign { name, op, value } => {
                let hir_value = self.lower_expr(value);
                let var_ty = self.var_type(name);
                let result_ty = self.infer_binop_type(op, &var_ty, &hir_value.ty);
                let ident = HirExpr::new(HirExprKind::Ident(name.clone()), var_ty);
                let rhs = if matches!(op, BinOp::Append) {
                    HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_push".to_string(),
                            args: vec![ident, hir_value],
                        },
                        result_ty.clone(),
                    )
                } else {
                    HirExpr::new(
                        HirExprKind::BinOp {
                            op: self.lower_binop(op),
                            left: Box::new(ident),
                            right: Box::new(hir_value),
                        },
                        result_ty.clone(),
                    )
                };
                out.push(HirStmt::Assign {
                    name: name.clone(),
                    ty: result_ty,
                    value: rhs,
                });
            }

            // Desugar: arr[i] += 1 -> arr[i] = arr[i] + 1
            Stmt::CompoundIndexAssign {
                target,
                index,
                op,
                value,
            } => {
                let hir_target = self.lower_expr(target);
                let hir_index = self.lower_expr(index);
                let hir_value = self.lower_expr(value);

                let current = HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(hir_target.clone()),
                        index: Box::new(hir_index.clone()),
                    },
                    Type::Any,
                );
                let result_ty = self.infer_binop_type(op, &current.ty, &hir_value.ty);
                let rhs = if matches!(op, BinOp::Append) {
                    HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_push".to_string(),
                            args: vec![current, hir_value],
                        },
                        result_ty,
                    )
                } else {
                    HirExpr::new(
                        HirExprKind::BinOp {
                            op: self.lower_binop(op),
                            left: Box::new(current),
                            right: Box::new(hir_value),
                        },
                        result_ty,
                    )
                };
                out.push(HirStmt::IndexAssign {
                    target: hir_target,
                    index: hir_index,
                    value: rhs,
                });
            }

            // Desugar: m.x += 1 -> m.x = m.x + 1
            Stmt::CompoundMemberAssign {
                target,
                field,
                op,
                value,
            } => {
                let hir_target = self.lower_expr(target);
                let hir_value = self.lower_expr(value);

                let current = HirExpr::new(
                    HirExprKind::Member {
                        target: Box::new(hir_target.clone()),
                        field: field.clone(),
                    },
                    Type::Any,
                );
                let result_ty = self.infer_binop_type(op, &current.ty, &hir_value.ty);
                let rhs = if matches!(op, BinOp::Append) {
                    HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_push".to_string(),
                            args: vec![current, hir_value],
                        },
                        result_ty,
                    )
                } else {
                    HirExpr::new(
                        HirExprKind::BinOp {
                            op: self.lower_binop(op),
                            left: Box::new(current),
                            right: Box::new(hir_value),
                        },
                        result_ty,
                    )
                };
                out.push(HirStmt::MemberAssign {
                    target: hir_target,
                    field: field.clone(),
                    value: rhs,
                });
            }

            // Desugar: a b = expr -> _tmp = expr; a = _tmp.0; b = _tmp.1
            Stmt::TupleDestructure { names, value } => {
                let hir_value = self.lower_expr(value);
                let val_ty = hir_value.ty.clone();
                let tmp = self.gensym();

                out.push(HirStmt::Assign {
                    name: tmp.clone(),
                    ty: val_ty.clone(),
                    value: hir_value,
                });

                for (i, name) in names.iter().enumerate() {
                    let elem_ty = match &val_ty {
                        Type::Tuple(elts) => elts.get(i).cloned().unwrap_or(Type::Any),
                        _ => Type::Any,
                    };
                    let index_expr = HirExpr::new(
                        HirExprKind::Index {
                            target: Box::new(HirExpr::new(
                                HirExprKind::Ident(tmp.clone()),
                                val_ty.clone(),
                            )),
                            index: Box::new(HirExpr::new(HirExprKind::Int(i as i64), Type::Int)),
                        },
                        elem_ty.clone(),
                    );
                    out.push(HirStmt::Assign {
                        name: name.clone(),
                        ty: elem_ty,
                        value: index_expr,
                    });
                }
            }

            // Desugar: {a b} = expr -> _tmp = expr; a = _tmp.a; b = _tmp.b
            Stmt::MapDestructure { names, value } => {
                let hir_value = self.lower_expr(value);
                let val_ty = hir_value.ty.clone();
                let tmp = self.gensym();

                out.push(HirStmt::Assign {
                    name: tmp.clone(),
                    ty: val_ty.clone(),
                    value: hir_value,
                });

                let elem_ty = match &val_ty {
                    Type::Map(inner) => *inner.clone(),
                    _ => Type::Any,
                };

                for name in names {
                    let member_expr = HirExpr::new(
                        HirExprKind::Member {
                            target: Box::new(HirExpr::new(
                                HirExprKind::Ident(tmp.clone()),
                                val_ty.clone(),
                            )),
                            field: name.clone(),
                        },
                        elem_ty.clone(),
                    );
                    out.push(HirStmt::Assign {
                        name: name.clone(),
                        ty: elem_ty.clone(),
                        value: member_expr,
                    });
                }
            }

            // Desugar: [h ..t] = expr -> _tmp = expr; h = _tmp[0]; t = slice(_tmp, 1, len(_tmp))
            Stmt::ArrayDestructure { head, tail, value } => {
                let hir_value = self.lower_expr(value);
                let val_ty = hir_value.ty.clone();
                let tmp = self.gensym();

                out.push(HirStmt::Assign {
                    name: tmp.clone(),
                    ty: val_ty.clone(),
                    value: hir_value,
                });

                let elem_ty = match &val_ty {
                    Type::Array(inner) => *inner.clone(),
                    _ => Type::Any,
                };

                // h = _tmp[0]
                let head_expr = HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(HirExpr::new(
                            HirExprKind::Ident(tmp.clone()),
                            val_ty.clone(),
                        )),
                        index: Box::new(HirExpr::new(HirExprKind::Int(0), Type::Int)),
                    },
                    elem_ty.clone(),
                );
                out.push(HirStmt::Assign {
                    name: head.clone(),
                    ty: elem_ty,
                    value: head_expr,
                });

                // t = tok_array_slice(_tmp, 1, #_tmp)
                let tmp_ident = HirExpr::new(HirExprKind::Ident(tmp.clone()), val_ty.clone());
                let tail_expr = HirExpr::new(
                    HirExprKind::RuntimeCall {
                        name: "tok_array_slice".to_string(),
                        args: vec![
                            tmp_ident.clone(),
                            HirExpr::new(HirExprKind::Int(1), Type::Int),
                            HirExpr::new(HirExprKind::Length(Box::new(tmp_ident)), Type::Int),
                        ],
                    },
                    val_ty,
                );
                out.push(HirStmt::Assign {
                    name: tail.clone(),
                    ty: tail_expr.ty.clone(),
                    value: tail_expr,
                });
            }

            Stmt::Import(path) => {
                out.push(HirStmt::Import(path.clone()));
            }

            Stmt::Return(expr) => {
                let hir_expr = expr.as_ref().map(|e| self.lower_expr(e));
                out.push(HirStmt::Return(hir_expr));
            }

            Stmt::Break => {
                out.push(HirStmt::Break);
            }

            Stmt::Continue => {
                out.push(HirStmt::Continue);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Expression lowering
    // ═══════════════════════════════════════════════════════════

    fn lower_expr(&mut self, expr: &Expr) -> HirExpr {
        match expr {
            // Literals -- direct mapping
            Expr::Int(v) => HirExpr::new(HirExprKind::Int(*v), Type::Int),
            Expr::Float(v) => HirExpr::new(HirExprKind::Float(*v), Type::Float),
            Expr::Str(v) => HirExpr::new(HirExprKind::Str(v.clone()), Type::Str),
            Expr::Bool(v) => HirExpr::new(HirExprKind::Bool(*v), Type::Bool),
            Expr::Nil => HirExpr::new(HirExprKind::Nil, Type::Nil),

            // Identifiers
            Expr::Ident(name) => {
                let ty = self.var_type(name);
                HirExpr::new(HirExprKind::Ident(name.clone()), ty)
            }

            // Desugar string interpolation:
            // "hello {name}!" -> tok_string_concat(tok_string_concat("hello ", tok_value_to_string(name)), "!")
            Expr::Interp(parts) => self.lower_interp(parts),

            // Compound literals
            Expr::Array(elts) => self.lower_array(elts),

            Expr::Map(pairs) => {
                let hir_pairs: Vec<(String, HirExpr)> = pairs
                    .iter()
                    .map(|(key, val)| {
                        let key_str = match key {
                            MapKey::Ident(s) | MapKey::Str(s) => s.clone(),
                        };
                        (key_str, self.lower_expr(val))
                    })
                    .collect();
                let val_ty = if hir_pairs.is_empty() {
                    Type::Any
                } else {
                    let mut t = hir_pairs[0].1.ty.clone();
                    for (_, v) in &hir_pairs[1..] {
                        t = tok_types::unify(&t, &v.ty);
                    }
                    t
                };
                HirExpr::new(HirExprKind::Map(hir_pairs), Type::Map(Box::new(val_ty)))
            }

            Expr::Tuple(elts) => {
                let hir_elts: Vec<HirExpr> = elts.iter().map(|e| self.lower_expr(e)).collect();
                let tys: Vec<Type> = hir_elts.iter().map(|e| e.ty.clone()).collect();
                HirExpr::new(HirExprKind::Tuple(hir_elts), Type::Tuple(tys))
            }

            // Range
            Expr::Range {
                start,
                end,
                inclusive,
            } => {
                let hir_start = self.lower_expr(start);
                let hir_end = self.lower_expr(end);
                HirExpr::new(
                    HirExprKind::Range {
                        start: Box::new(hir_start),
                        end: Box::new(hir_end),
                        inclusive: *inclusive,
                    },
                    Type::Range,
                )
            }

            // Binary ops
            Expr::BinOp { op, left, right } => {
                if matches!(op, BinOp::Append) {
                    let hir_left = self.lower_expr(left);
                    let hir_right = self.lower_expr(right);
                    let ty = self.infer_binop_type(op, &hir_left.ty, &hir_right.ty);
                    return HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_push".to_string(),
                            args: vec![hir_left, hir_right],
                        },
                        ty,
                    );
                }
                let hir_left = self.lower_expr(left);
                let hir_right = self.lower_expr(right);
                let ty = self.infer_binop_type(op, &hir_left.ty, &hir_right.ty);
                HirExpr::new(
                    HirExprKind::BinOp {
                        op: self.lower_binop(op),
                        left: Box::new(hir_left),
                        right: Box::new(hir_right),
                    },
                    ty,
                )
            }

            // Unary ops
            Expr::UnaryOp { op, expr } => {
                let hir_expr = self.lower_expr(expr);
                let ty = match op {
                    UnaryOp::Neg => match &hir_expr.ty {
                        Type::Int => Type::Int,
                        Type::Float => Type::Float,
                        _ => Type::Any,
                    },
                    UnaryOp::Not => Type::Bool,
                };
                HirExpr::new(
                    HirExprKind::UnaryOp {
                        op: self.lower_unaryop(op),
                        operand: Box::new(hir_expr),
                    },
                    ty,
                )
            }

            // Index
            Expr::Index {
                expr: target,
                index,
            } => {
                let hir_target = self.lower_expr(target);
                let hir_index = self.lower_expr(index);
                let ty = match &hir_target.ty {
                    Type::Array(inner) => *inner.clone(),
                    Type::Map(inner) => *inner.clone(),
                    Type::Tuple(elts) => {
                        if let HirExprKind::Int(i) = &hir_index.kind {
                            elts.get(*i as usize).cloned().unwrap_or(Type::Any)
                        } else {
                            Type::Any
                        }
                    }
                    Type::Str => Type::Str,
                    _ => Type::Any,
                };
                HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(hir_target),
                        index: Box::new(hir_index),
                    },
                    ty,
                )
            }

            // Member
            Expr::Member {
                expr: target,
                field,
            } => {
                let hir_target = self.lower_expr(target);
                let ty = self.infer_member_type(&hir_target.ty, field);
                HirExpr::new(
                    HirExprKind::Member {
                        target: Box::new(hir_target),
                        field: field.clone(),
                    },
                    ty,
                )
            }

            // Desugar optional chain: expr.?field ->
            //   if expr != Nil { expr.field } else { Nil }
            Expr::OptionalChain {
                expr: target,
                field,
            } => {
                let hir_target = self.lower_expr(target);
                let target_ty = hir_target.ty.clone();
                let field_ty = self.infer_member_type(&target_ty, field);
                let result_ty = Type::Optional(Box::new(field_ty.clone()));

                let tmp = self.gensym();
                // Build: if _tmp != Nil then _tmp.field else Nil
                let cond = HirExpr::new(
                    HirExprKind::BinOp {
                        op: HirBinOp::Neq,
                        left: Box::new(HirExpr::new(
                            HirExprKind::Ident(tmp.clone()),
                            target_ty.clone(),
                        )),
                        right: Box::new(HirExpr::new(HirExprKind::Nil, Type::Nil)),
                    },
                    Type::Bool,
                );
                let then_expr = HirExpr::new(
                    HirExprKind::Member {
                        target: Box::new(HirExpr::new(
                            HirExprKind::Ident(tmp.clone()),
                            target_ty.clone(),
                        )),
                        field: field.clone(),
                    },
                    field_ty,
                );

                // Wrap in a block: { _tmp = target; if _tmp != Nil ... }
                HirExpr::new(
                    HirExprKind::Block {
                        stmts: vec![HirStmt::Assign {
                            name: tmp,
                            ty: target_ty,
                            value: hir_target,
                        }],
                        expr: Some(Box::new(HirExpr::new(
                            HirExprKind::If {
                                cond: Box::new(cond),
                                then_body: vec![],
                                then_expr: Some(Box::new(then_expr)),
                                else_body: vec![],
                                else_expr: Some(Box::new(HirExpr::new(
                                    HirExprKind::Nil,
                                    Type::Nil,
                                ))),
                            },
                            result_ty.clone(),
                        ))),
                    },
                    result_ty,
                )
            }

            // Function call
            Expr::Call { func, args } => {
                let hir_func = self.lower_expr(func);
                let hir_args: Vec<HirExpr> = args.iter().map(|a| self.lower_expr(a)).collect();
                let ret_ty = match &hir_func.ty {
                    Type::Func(ft) => *ft.ret.clone(),
                    _ => Type::Any,
                };
                HirExpr::new(
                    HirExprKind::Call {
                        func: Box::new(hir_func),
                        args: hir_args,
                    },
                    ret_ty,
                )
            }

            // Lambda
            Expr::Lambda {
                params,
                ret_type: _,
                body,
            } => {
                let hir_params = self.lower_params(params);
                self.push_scope();
                for hp in &hir_params {
                    self.define_local(&hp.name, hp.ty.clone());
                }
                let hir_body = self.lower_func_body(body);
                self.pop_scope();
                let param_types: Vec<tok_types::ParamType> = hir_params
                    .iter()
                    .map(|p| tok_types::ParamType {
                        ty: p.ty.clone(),
                        has_default: false,
                    })
                    .collect();
                let ret_ty = Type::Any; // simplified
                let func_ty = Type::Func(tok_types::FuncType {
                    params: param_types,
                    ret: Box::new(ret_ty.clone()),
                    variadic: false,
                });
                HirExpr::new(
                    HirExprKind::Lambda {
                        params: hir_params,
                        ret_type: ret_ty,
                        body: hir_body,
                    },
                    func_ty,
                )
            }

            // Ternary -> If
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                let hir_cond = self.lower_expr(cond);
                let hir_then = self.lower_expr(then_expr);
                let then_ty = hir_then.ty.clone();
                let (else_body, hir_else, result_ty) = if let Some(else_e) = else_expr {
                    let hir_else = self.lower_expr(else_e);
                    let ty = tok_types::unify(&then_ty, &hir_else.ty);
                    (vec![], Some(Box::new(hir_else)), ty)
                } else {
                    (vec![], None, then_ty)
                };
                HirExpr::new(
                    HirExprKind::If {
                        cond: Box::new(hir_cond),
                        then_body: vec![],
                        then_expr: Some(Box::new(hir_then)),
                        else_body,
                        else_expr: hir_else,
                    },
                    result_ty,
                )
            }

            // Desugar Match -> if-else chain
            Expr::Match { subject, arms } => self.lower_match(subject, arms),

            // Loop
            Expr::Loop { clause, body } => self.lower_loop(clause, body),

            // Block
            Expr::Block(stmts) => self.lower_block(stmts),

            // Desugar pipeline: x |> f -> f(x)
            //                   x |> f(y) -> f(x, y)
            Expr::Pipeline { left, right } => self.lower_pipeline(left, right),

            // Desugar filter: arr ?> pred -> RuntimeCall("tok_array_filter", [arr, pred])
            Expr::Filter { expr, pred } => {
                let hir_arr = self.lower_expr(expr);
                let arr_ty = hir_arr.ty.clone();
                let hir_pred = self.lower_expr(pred);
                HirExpr::new(
                    HirExprKind::RuntimeCall {
                        name: "tok_array_filter".to_string(),
                        args: vec![hir_arr, hir_pred],
                    },
                    arr_ty,
                )
            }

            // Desugar reduce: arr /> init fn -> RuntimeCall("tok_array_reduce", [arr, init, fn])
            Expr::Reduce { expr, init, func } => {
                let hir_arr = self.lower_expr(expr);
                let hir_func = self.lower_expr(func);
                let result_ty = if let Some(init_e) = init {
                    self.infer_expr_type(init_e)
                } else {
                    match &hir_arr.ty {
                        Type::Array(inner) => *inner.clone(),
                        _ => Type::Any,
                    }
                };
                let hir_init = if let Some(init_e) = init {
                    self.lower_expr(init_e)
                } else {
                    // When no init, pass Nil as sentinel (runtime handles first-element init)
                    HirExpr::new(HirExprKind::Nil, Type::Nil)
                };
                HirExpr::new(
                    HirExprKind::RuntimeCall {
                        name: "tok_array_reduce".to_string(),
                        args: vec![hir_arr, hir_init, hir_func],
                    },
                    result_ty,
                )
            }

            // Spread -- should only appear inside arrays, handled in lower_array
            Expr::Spread(inner) => {
                // If encountered outside an array context, just lower the inner expression
                self.lower_expr(inner)
            }

            // Length
            Expr::Length(inner) => {
                let hir_inner = self.lower_expr(inner);
                HirExpr::new(HirExprKind::Length(Box::new(hir_inner)), Type::Int)
            }

            // Desugar nil coalesce: left ?? right -> if left != Nil { left } else { right }
            Expr::NilCoalesce { left, right } => {
                let hir_left = self.lower_expr(left);
                let hir_right = self.lower_expr(right);
                let left_ty = hir_left.ty.clone();
                let result_ty = match &left_ty {
                    Type::Optional(inner) => tok_types::unify(inner, &hir_right.ty),
                    Type::Nil => hir_right.ty.clone(),
                    _ => tok_types::unify(&left_ty, &hir_right.ty),
                };

                let tmp = self.gensym();
                HirExpr::new(
                    HirExprKind::Block {
                        stmts: vec![HirStmt::Assign {
                            name: tmp.clone(),
                            ty: left_ty.clone(),
                            value: hir_left,
                        }],
                        expr: Some(Box::new(HirExpr::new(
                            HirExprKind::If {
                                cond: Box::new(HirExpr::new(
                                    HirExprKind::BinOp {
                                        op: HirBinOp::Neq,
                                        left: Box::new(HirExpr::new(
                                            HirExprKind::Ident(tmp.clone()),
                                            left_ty.clone(),
                                        )),
                                        right: Box::new(HirExpr::new(HirExprKind::Nil, Type::Nil)),
                                    },
                                    Type::Bool,
                                )),
                                then_body: vec![],
                                then_expr: Some(Box::new(HirExpr::new(
                                    HirExprKind::Ident(tmp),
                                    left_ty,
                                ))),
                                else_body: vec![],
                                else_expr: Some(Box::new(hir_right)),
                            },
                            result_ty.clone(),
                        ))),
                    },
                    result_ty,
                )
            }

            // Desugar error propagation: expr?^ ->
            //   _tmp = expr; if _tmp.1 != Nil { return _tmp } else { _tmp.0 }
            Expr::ErrorPropagate(inner) => {
                let hir_inner = self.lower_expr(inner);
                let inner_ty = hir_inner.ty.clone();
                let ok_ty = match &inner_ty {
                    Type::Result(ok_ty) => *ok_ty.clone(),
                    Type::Tuple(elts) if elts.len() == 2 => elts[0].clone(),
                    _ => Type::Any,
                };

                let tmp = self.gensym();
                let tmp_ident = HirExpr::new(HirExprKind::Ident(tmp.clone()), inner_ty.clone());

                // _tmp.1 (error field)
                let err_field = HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(tmp_ident.clone()),
                        index: Box::new(HirExpr::new(HirExprKind::Int(1), Type::Int)),
                    },
                    Type::Any,
                );

                // _tmp.0 (ok field)
                let ok_field = HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(tmp_ident.clone()),
                        index: Box::new(HirExpr::new(HirExprKind::Int(0), Type::Int)),
                    },
                    ok_ty.clone(),
                );

                // if _tmp.1 != Nil then return _tmp else _tmp.0
                HirExpr::new(
                    HirExprKind::Block {
                        stmts: vec![HirStmt::Assign {
                            name: tmp,
                            ty: inner_ty,
                            value: hir_inner,
                        }],
                        expr: Some(Box::new(HirExpr::new(
                            HirExprKind::If {
                                cond: Box::new(HirExpr::new(
                                    HirExprKind::BinOp {
                                        op: HirBinOp::Neq,
                                        left: Box::new(err_field),
                                        right: Box::new(HirExpr::new(HirExprKind::Nil, Type::Nil)),
                                    },
                                    Type::Bool,
                                )),
                                then_body: vec![HirStmt::Return(Some(tmp_ident))],
                                then_expr: None,
                                else_body: vec![],
                                else_expr: Some(Box::new(ok_field)),
                            },
                            ok_ty.clone(),
                        ))),
                    },
                    ok_ty,
                )
            }

            // Desugar conditional return: cond?^value -> if cond { return value }
            Expr::ConditionalReturn { cond, value } => {
                let hir_cond = self.lower_expr(cond);
                let hir_value = self.lower_expr(value);
                HirExpr::new(
                    HirExprKind::If {
                        cond: Box::new(hir_cond),
                        then_body: vec![HirStmt::Return(Some(hir_value))],
                        then_expr: None,
                        else_body: vec![],
                        else_expr: None,
                    },
                    Type::Never,
                )
            }

            // Concurrency
            Expr::Go(body) => {
                let hir_body = self.lower_expr(body);
                let ty = Type::Handle(Box::new(hir_body.ty.clone()));
                HirExpr::new(HirExprKind::Go(Box::new(hir_body)), ty)
            }

            Expr::Receive(inner) => {
                let hir_inner = self.lower_expr(inner);
                let ty = match &hir_inner.ty {
                    Type::Channel(t) => *t.clone(),
                    Type::Handle(t) => *t.clone(),
                    _ => Type::Any,
                };
                HirExpr::new(HirExprKind::Receive(Box::new(hir_inner)), ty)
            }

            Expr::Send { chan, value } => {
                let hir_chan = self.lower_expr(chan);
                let hir_value = self.lower_expr(value);
                HirExpr::new(
                    HirExprKind::Send {
                        chan: Box::new(hir_chan),
                        value: Box::new(hir_value),
                    },
                    Type::Nil,
                )
            }

            Expr::Select(arms) => {
                let hir_arms: Vec<HirSelectArm> = arms
                    .iter()
                    .map(|arm| match arm {
                        SelectArm::Recv { var, chan, body } => {
                            let hir_chan = self.lower_expr(chan);
                            let mut hir_body = Vec::new();
                            for s in body {
                                self.lower_stmt(s, &mut hir_body);
                            }
                            HirSelectArm::Recv {
                                var: var.clone(),
                                chan: hir_chan,
                                body: hir_body,
                            }
                        }
                        SelectArm::Send { chan, value, body } => {
                            let hir_chan = self.lower_expr(chan);
                            let hir_value = self.lower_expr(value);
                            let mut hir_body = Vec::new();
                            for s in body {
                                self.lower_stmt(s, &mut hir_body);
                            }
                            HirSelectArm::Send {
                                chan: hir_chan,
                                value: hir_value,
                                body: hir_body,
                            }
                        }
                        SelectArm::Default(body) => {
                            let mut hir_body = Vec::new();
                            for s in body {
                                self.lower_stmt(s, &mut hir_body);
                            }
                            HirSelectArm::Default(hir_body)
                        }
                    })
                    .collect();
                HirExpr::new(HirExprKind::Select(hir_arms), Type::Any)
            }

            // Import as expression
            Expr::Import(path) => {
                // Import as expression returns a map
                HirExpr::new(
                    HirExprKind::RuntimeCall {
                        name: "tok_import".to_string(),
                        args: vec![HirExpr::new(HirExprKind::Str(path.clone()), Type::Str)],
                    },
                    Type::Map(Box::new(Type::Any)),
                )
            }

            // Return as expression
            Expr::Return(expr) => {
                let hir_expr = expr.as_ref().map(|e| self.lower_expr(e));
                // Emit return as a block with a return statement and Never result
                HirExpr::new(
                    HirExprKind::Block {
                        stmts: vec![HirStmt::Return(hir_expr)],
                        expr: None,
                    },
                    Type::Never,
                )
            }

            Expr::Break => HirExpr::new(
                HirExprKind::Block {
                    stmts: vec![HirStmt::Break],
                    expr: None,
                },
                Type::Never,
            ),

            Expr::Continue => HirExpr::new(
                HirExprKind::Block {
                    stmts: vec![HirStmt::Continue],
                    expr: None,
                },
                Type::Never,
            ),
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Complex desugaring helpers
    // ═══════════════════════════════════════════════════════════

    /// Lower string interpolation to a chain of tok_string_concat runtime calls.
    fn lower_interp(&mut self, parts: &[InterpPart]) -> HirExpr {
        let mut result: Option<HirExpr> = None;

        for part in parts {
            let segment = match part {
                InterpPart::Lit(s) => {
                    if s.is_empty() {
                        continue;
                    }
                    HirExpr::new(HirExprKind::Str(s.clone()), Type::Str)
                }
                InterpPart::Expr(e) => {
                    let hir_e = self.lower_expr(e);
                    // If the expression is already a string, use it directly;
                    // otherwise wrap with tok_value_to_string
                    if matches!(hir_e.ty, Type::Str) {
                        hir_e
                    } else {
                        // Choose a specific conversion based on known type
                        let conv_name = match &hir_e.ty {
                            Type::Int => "tok_int_to_string",
                            Type::Float => "tok_float_to_string",
                            Type::Bool => "tok_bool_to_string",
                            _ => "tok_value_to_string",
                        };
                        HirExpr::new(
                            HirExprKind::RuntimeCall {
                                name: conv_name.to_string(),
                                args: vec![hir_e],
                            },
                            Type::Str,
                        )
                    }
                }
            };

            result = Some(match result {
                None => segment,
                Some(acc) => HirExpr::new(
                    HirExprKind::RuntimeCall {
                        name: "tok_string_concat".to_string(),
                        args: vec![acc, segment],
                    },
                    Type::Str,
                ),
            });
        }

        // If all parts were empty, return empty string
        result.unwrap_or_else(|| HirExpr::new(HirExprKind::Str(String::new()), Type::Str))
    }

    /// Lower array literal, handling spread elements.
    fn lower_array(&mut self, elts: &[Expr]) -> HirExpr {
        let has_spread = elts.iter().any(|e| matches!(e, Expr::Spread(_)));

        if !has_spread {
            // Simple case: no spreads
            let hir_elts: Vec<HirExpr> = elts.iter().map(|e| self.lower_expr(e)).collect();
            let elem_ty = if hir_elts.is_empty() {
                Type::Any
            } else {
                let mut t = hir_elts[0].ty.clone();
                for e in &hir_elts[1..] {
                    t = tok_types::unify(&t, &e.ty);
                }
                t
            };
            return HirExpr::new(HirExprKind::Array(hir_elts), Type::Array(Box::new(elem_ty)));
        }

        // Has spreads: desugar to alloc + concat/push chain
        let tmp = self.gensym();
        let arr_ty = Type::Array(Box::new(Type::Any));

        let mut stmts = Vec::new();
        // _tmp = tok_array_alloc()
        stmts.push(HirStmt::Assign {
            name: tmp.clone(),
            ty: arr_ty.clone(),
            value: HirExpr::new(
                HirExprKind::RuntimeCall {
                    name: "tok_array_alloc".to_string(),
                    args: vec![],
                },
                arr_ty.clone(),
            ),
        });

        for elt in elts {
            match elt {
                Expr::Spread(inner) => {
                    let hir_inner = self.lower_expr(inner);
                    // _tmp = tok_array_concat(_tmp, inner)
                    stmts.push(HirStmt::Assign {
                        name: tmp.clone(),
                        ty: arr_ty.clone(),
                        value: HirExpr::new(
                            HirExprKind::RuntimeCall {
                                name: "tok_array_concat".to_string(),
                                args: vec![
                                    HirExpr::new(HirExprKind::Ident(tmp.clone()), arr_ty.clone()),
                                    hir_inner,
                                ],
                            },
                            arr_ty.clone(),
                        ),
                    });
                }
                _ => {
                    let hir_elt = self.lower_expr(elt);
                    // _tmp = tok_array_push(_tmp, elt)
                    stmts.push(HirStmt::Assign {
                        name: tmp.clone(),
                        ty: arr_ty.clone(),
                        value: HirExpr::new(
                            HirExprKind::RuntimeCall {
                                name: "tok_array_push".to_string(),
                                args: vec![
                                    HirExpr::new(HirExprKind::Ident(tmp.clone()), arr_ty.clone()),
                                    hir_elt,
                                ],
                            },
                            arr_ty.clone(),
                        ),
                    });
                }
            }
        }

        HirExpr::new(
            HirExprKind::Block {
                stmts,
                expr: Some(Box::new(HirExpr::new(
                    HirExprKind::Ident(tmp),
                    arr_ty.clone(),
                ))),
            },
            arr_ty,
        )
    }

    /// Lower pipeline: x |> f -> f(x); x |> f(y) -> f(x, y)
    fn lower_pipeline(&mut self, left: &Expr, right: &Expr) -> HirExpr {
        let hir_left = self.lower_expr(left);

        match right {
            // x |> f(y, z) -> f(x, y, z)
            Expr::Call { func, args } => {
                let hir_func = self.lower_expr(func);
                let mut hir_args = vec![hir_left];
                for arg in args {
                    hir_args.push(self.lower_expr(arg));
                }
                let ret_ty = match &hir_func.ty {
                    Type::Func(ft) => *ft.ret.clone(),
                    _ => Type::Any,
                };
                HirExpr::new(
                    HirExprKind::Call {
                        func: Box::new(hir_func),
                        args: hir_args,
                    },
                    ret_ty,
                )
            }
            // x |> f -> f(x)
            _ => {
                let hir_func = self.lower_expr(right);
                let ret_ty = match &hir_func.ty {
                    Type::Func(ft) => *ft.ret.clone(),
                    _ => Type::Any,
                };
                HirExpr::new(
                    HirExprKind::Call {
                        func: Box::new(hir_func),
                        args: vec![hir_left],
                    },
                    ret_ty,
                )
            }
        }
    }

    /// Lower match to if-else chain.
    fn lower_match(&mut self, subject: &Option<Box<Expr>>, arms: &[ast::MatchArm]) -> HirExpr {
        let hir_subject = subject.as_ref().map(|s| self.lower_expr(s));
        let (tmp_name, tmp_stmts) = if let Some(ref subj) = hir_subject {
            let tmp = self.gensym();
            let stmt = HirStmt::Assign {
                name: tmp.clone(),
                ty: subj.ty.clone(),
                value: subj.clone(),
            };
            (Some(tmp), vec![stmt])
        } else {
            (None, vec![])
        };

        let subject_ty = hir_subject
            .as_ref()
            .map(|s| s.ty.clone())
            .unwrap_or(Type::Any);

        let if_chain = self.lower_match_arms(arms, &tmp_name, &subject_ty);

        if tmp_stmts.is_empty() {
            if_chain
        } else {
            let result_ty = if_chain.ty.clone();
            HirExpr::new(
                HirExprKind::Block {
                    stmts: tmp_stmts,
                    expr: Some(Box::new(if_chain)),
                },
                result_ty,
            )
        }
    }

    fn lower_match_arms(
        &mut self,
        arms: &[ast::MatchArm],
        subject_tmp: &Option<String>,
        subject_ty: &Type,
    ) -> HirExpr {
        if arms.is_empty() {
            return HirExpr::new(HirExprKind::Nil, Type::Nil);
        }

        let arm = &arms[0];
        let rest = &arms[1..];

        let body_expr = self.lower_match_body(&arm.body);
        let body_ty = body_expr.ty.clone();

        match &arm.pattern {
            Pattern::Wildcard => {
                // Wildcard matches everything -- this is the final else
                body_expr
            }
            Pattern::Guard(guard_expr) => {
                // Guard: condition is the guard expression itself
                let cond = self.lower_expr(guard_expr);
                let else_expr = self.lower_match_arms(rest, subject_tmp, subject_ty);
                let result_ty = tok_types::unify(&body_ty, &else_expr.ty);
                HirExpr::new(
                    HirExprKind::If {
                        cond: Box::new(cond),
                        then_body: vec![],
                        then_expr: Some(Box::new(body_expr)),
                        else_body: vec![],
                        else_expr: Some(Box::new(else_expr)),
                    },
                    result_ty,
                )
            }
            _ => {
                // Value pattern: compare subject with pattern value
                let pat_expr = self.pattern_to_expr(&arm.pattern);
                let cond = if let Some(ref tmp) = subject_tmp {
                    HirExpr::new(
                        HirExprKind::BinOp {
                            op: HirBinOp::Eq,
                            left: Box::new(HirExpr::new(
                                HirExprKind::Ident(tmp.clone()),
                                subject_ty.clone(),
                            )),
                            right: Box::new(pat_expr),
                        },
                        Type::Bool,
                    )
                } else {
                    // No subject -- pattern itself is the condition (guard-like)
                    pat_expr
                };
                let else_expr = self.lower_match_arms(rest, subject_tmp, subject_ty);
                let result_ty = tok_types::unify(&body_ty, &else_expr.ty);
                HirExpr::new(
                    HirExprKind::If {
                        cond: Box::new(cond),
                        then_body: vec![],
                        then_expr: Some(Box::new(body_expr)),
                        else_body: vec![],
                        else_expr: Some(Box::new(else_expr)),
                    },
                    result_ty,
                )
            }
        }
    }

    fn pattern_to_expr(&self, pat: &Pattern) -> HirExpr {
        match pat {
            Pattern::Int(v) => HirExpr::new(HirExprKind::Int(*v), Type::Int),
            Pattern::Float(v) => HirExpr::new(HirExprKind::Float(*v), Type::Float),
            Pattern::Str(v) => HirExpr::new(HirExprKind::Str(v.clone()), Type::Str),
            Pattern::Bool(v) => HirExpr::new(HirExprKind::Bool(*v), Type::Bool),
            Pattern::Nil => HirExpr::new(HirExprKind::Nil, Type::Nil),
            Pattern::Ident(name) => {
                let ty = self.var_type(name);
                HirExpr::new(HirExprKind::Ident(name.clone()), ty)
            }
            Pattern::Wildcard => HirExpr::new(HirExprKind::Bool(true), Type::Bool),
            Pattern::Tuple(pats) => {
                let hir_elts: Vec<HirExpr> = pats.iter().map(|p| self.pattern_to_expr(p)).collect();
                let tys: Vec<Type> = hir_elts.iter().map(|e| e.ty.clone()).collect();
                HirExpr::new(HirExprKind::Tuple(hir_elts), Type::Tuple(tys))
            }
            Pattern::Guard(_expr) => {
                // Guards shouldn't appear here (handled separately), but just in case
                // We can't call lower_expr because it takes &mut self.
                // Return a placeholder.
                HirExpr::new(HirExprKind::Bool(true), Type::Bool)
            }
        }
    }

    fn lower_match_body(&mut self, body: &MatchBody) -> HirExpr {
        match body {
            MatchBody::Expr(e) => self.lower_expr(e),
            MatchBody::Block(stmts) => self.lower_block(stmts),
        }
    }

    /// Define loop variables in scope based on the loop clause.
    fn define_loop_vars(&mut self, clause: &LoopClause) {
        match clause {
            LoopClause::ForRange { var, .. } => {
                self.define_local(var, Type::Int);
            }
            LoopClause::ForEach { var, iter } => {
                let iter_ty = self.infer_expr_type(iter);
                let elem_ty = match iter_ty {
                    Type::Array(inner) => *inner,
                    Type::Str => Type::Str,
                    Type::Range => Type::Int,
                    _ => Type::Any,
                };
                self.define_local(var, elem_ty);
            }
            LoopClause::ForEachIndexed {
                idx_var,
                val_var,
                iter,
            } => {
                let iter_ty = self.infer_expr_type(iter);
                // For maps, idx is the string key; for arrays, idx is the integer index
                let idx_ty = if matches!(iter_ty, Type::Map(_)) {
                    Type::Str
                } else {
                    Type::Int
                };
                self.define_local(idx_var, idx_ty);
                let elem_ty = match iter_ty {
                    Type::Array(inner) => *inner,
                    Type::Map(inner) => *inner,
                    Type::Str => Type::Str,
                    Type::Range => Type::Int,
                    _ => Type::Any,
                };
                self.define_local(val_var, elem_ty);
            }
            LoopClause::While(_) | LoopClause::Infinite => {}
        }
    }

    /// Lower a loop expression.
    fn lower_loop(&mut self, clause: &LoopClause, body: &LoopBody) -> HirExpr {
        match body {
            LoopBody::Block(stmts) => {
                let hir_kind = self.lower_loop_clause(clause);
                self.push_scope();
                self.define_loop_vars(clause);
                let mut hir_body = Vec::new();
                for s in stmts {
                    self.lower_stmt(s, &mut hir_body);
                }
                self.pop_scope();
                HirExpr::new(
                    HirExprKind::Loop {
                        kind: Box::new(hir_kind),
                        body: hir_body,
                    },
                    Type::Nil,
                )
            }
            // Desugar collect loop: ~(i:0..n)=expr ->
            //   { _collect = []; loop { _collect = push(_collect, expr) }; _collect }
            LoopBody::Collect(collect_expr) => {
                self.push_scope();
                self.define_loop_vars(clause);
                let hir_kind = self.lower_loop_clause(clause);
                let collect_tmp = self.gensym();
                let elem_ty = self.infer_expr_type(collect_expr);
                let arr_ty = Type::Array(Box::new(elem_ty));

                let hir_collect_expr = self.lower_expr(collect_expr);

                let loop_body = vec![HirStmt::Assign {
                    name: collect_tmp.clone(),
                    ty: arr_ty.clone(),
                    value: HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_push".to_string(),
                            args: vec![
                                HirExpr::new(
                                    HirExprKind::Ident(collect_tmp.clone()),
                                    arr_ty.clone(),
                                ),
                                hir_collect_expr,
                            ],
                        },
                        arr_ty.clone(),
                    ),
                }];

                let init_stmt = HirStmt::Assign {
                    name: collect_tmp.clone(),
                    ty: arr_ty.clone(),
                    value: HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_alloc".to_string(),
                            args: vec![],
                        },
                        arr_ty.clone(),
                    ),
                };

                let loop_expr = HirExpr::new(
                    HirExprKind::Loop {
                        kind: Box::new(hir_kind),
                        body: loop_body,
                    },
                    Type::Nil,
                );

                self.pop_scope();

                HirExpr::new(
                    HirExprKind::Block {
                        stmts: vec![init_stmt, HirStmt::Expr(loop_expr)],
                        expr: Some(Box::new(HirExpr::new(
                            HirExprKind::Ident(collect_tmp),
                            arr_ty.clone(),
                        ))),
                    },
                    arr_ty,
                )
            }
        }
    }

    fn lower_loop_clause(&mut self, clause: &LoopClause) -> HirLoopKind {
        match clause {
            LoopClause::While(cond) => {
                let hir_cond = self.lower_expr(cond);
                HirLoopKind::While(Box::new(hir_cond))
            }
            LoopClause::ForRange { var, range } => {
                // The range expression is an Expr::Range { start, end, inclusive }
                match range {
                    Expr::Range {
                        start,
                        end,
                        inclusive,
                    } => {
                        let hir_start = self.lower_expr(start);
                        let hir_end = self.lower_expr(end);
                        HirLoopKind::ForRange {
                            var: var.clone(),
                            start: hir_start,
                            end: hir_end,
                            inclusive: *inclusive,
                        }
                    }
                    // Fallback: treat as foreach over a general iterable
                    other => {
                        let hir_iter = self.lower_expr(other);
                        HirLoopKind::ForEach {
                            var: var.clone(),
                            iter: hir_iter,
                        }
                    }
                }
            }
            LoopClause::ForEach { var, iter } => {
                let hir_iter = self.lower_expr(iter);
                HirLoopKind::ForEach {
                    var: var.clone(),
                    iter: hir_iter,
                }
            }
            LoopClause::ForEachIndexed {
                idx_var,
                val_var,
                iter,
            } => {
                let hir_iter = self.lower_expr(iter);
                HirLoopKind::ForEachIndexed {
                    idx_var: idx_var.clone(),
                    val_var: val_var.clone(),
                    iter: hir_iter,
                }
            }
            LoopClause::Infinite => HirLoopKind::Infinite,
        }
    }

    fn lower_block(&mut self, stmts: &[Stmt]) -> HirExpr {
        let mut hir_stmts = Vec::new();
        for s in stmts {
            self.lower_stmt(s, &mut hir_stmts);
        }
        // The last expression statement becomes the block's value
        let expr = if let Some(HirStmt::Expr(_)) = hir_stmts.last() {
            if let Some(HirStmt::Expr(e)) = hir_stmts.pop() {
                Some(Box::new(e))
            } else {
                None
            }
        } else {
            None
        };
        let ty = expr.as_ref().map(|e| e.ty.clone()).unwrap_or(Type::Nil);
        HirExpr::new(
            HirExprKind::Block {
                stmts: hir_stmts,
                expr,
            },
            ty,
        )
    }

    // ═══════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════

    fn lower_params(&self, params: &[Param]) -> Vec<HirParam> {
        params
            .iter()
            .map(|p| HirParam {
                name: p.name.clone(),
                ty: p
                    .ty
                    .as_ref()
                    .map(|te| self.resolve_type_expr(te))
                    .unwrap_or(Type::Any),
            })
            .collect()
    }

    #[allow(clippy::only_used_in_recursion)]
    fn resolve_type_expr(&self, te: &ast::TypeExpr) -> Type {
        match te {
            ast::TypeExpr::Prim(p) => match p {
                ast::PrimType::Int => Type::Int,
                ast::PrimType::Float => Type::Float,
                ast::PrimType::Str => Type::Str,
                ast::PrimType::Bool => Type::Bool,
                ast::PrimType::Nil => Type::Nil,
                ast::PrimType::Any => Type::Any,
            },
            ast::TypeExpr::Array(inner) => Type::Array(Box::new(self.resolve_type_expr(inner))),
            ast::TypeExpr::Map(inner) => Type::Map(Box::new(self.resolve_type_expr(inner))),
            ast::TypeExpr::Tuple(elts) => {
                Type::Tuple(elts.iter().map(|e| self.resolve_type_expr(e)).collect())
            }
            ast::TypeExpr::Func(params, ret) => {
                let pts = params
                    .iter()
                    .map(|p| tok_types::ParamType {
                        ty: self.resolve_type_expr(p),
                        has_default: false,
                    })
                    .collect();
                Type::Func(tok_types::FuncType {
                    params: pts,
                    ret: Box::new(self.resolve_type_expr(ret)),
                    variadic: false,
                })
            }
            ast::TypeExpr::Optional(inner) => {
                Type::Optional(Box::new(self.resolve_type_expr(inner)))
            }
            ast::TypeExpr::Result(inner) => Type::Result(Box::new(self.resolve_type_expr(inner))),
            ast::TypeExpr::Channel(inner) => Type::Channel(Box::new(self.resolve_type_expr(inner))),
            ast::TypeExpr::Handle(inner) => Type::Handle(Box::new(self.resolve_type_expr(inner))),
            ast::TypeExpr::Var(_) => Type::Any,
        }
    }

    fn lower_func_body(&mut self, body: &FuncBody) -> Vec<HirStmt> {
        match body {
            FuncBody::Expr(expr) => {
                let hir_expr = self.lower_expr(expr);
                vec![HirStmt::Return(Some(hir_expr))]
            }
            FuncBody::Block(stmts) => {
                let mut out = Vec::new();
                for s in stmts {
                    self.lower_stmt(s, &mut out);
                }
                out
            }
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
}
