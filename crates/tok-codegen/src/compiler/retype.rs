// ─── HIR type rewriting for specialized lambda bodies ──────────────────

use std::collections::HashMap;

use tok_hir::hir::*;
use tok_types::Type;

/// Convert Return(Some(expr)) → Expr(expr) in a statement list.
/// Used when inlining a lambda body: the lambda's "return" should produce a value
/// in the current block, not jump to the enclosing function's return block.
pub(crate) fn unwrap_return_stmts(stmts: Vec<HirStmt>) -> Vec<HirStmt> {
    stmts
        .into_iter()
        .map(|s| match s {
            HirStmt::Return(Some(expr)) => HirStmt::Expr(expr),
            HirStmt::Return(None) => HirStmt::Expr(HirExpr {
                kind: HirExprKind::Nil,
                ty: Type::Nil,
            }),
            other => other,
        })
        .collect()
}

pub(crate) fn retype_body(body: &[HirStmt], type_map: &HashMap<String, Type>) -> Vec<HirStmt> {
    body.iter().map(|s| retype_stmt(s, type_map)).collect()
}

fn retype_stmt(stmt: &HirStmt, type_map: &HashMap<String, Type>) -> HirStmt {
    match stmt {
        HirStmt::Expr(e) => HirStmt::Expr(retype_expr(e, type_map)),
        HirStmt::Assign { name, ty, value } => {
            let new_value = retype_expr(value, type_map);
            HirStmt::Assign {
                name: name.clone(),
                ty: if matches!(ty, Type::Any) {
                    new_value.ty.clone()
                } else {
                    ty.clone()
                },
                value: new_value,
            }
        }
        HirStmt::Return(Some(e)) => HirStmt::Return(Some(retype_expr(e, type_map))),
        other => other.clone(),
    }
}

pub(crate) fn retype_expr(expr: &HirExpr, type_map: &HashMap<String, Type>) -> HirExpr {
    let mut e = expr.clone();
    match &mut e.kind {
        HirExprKind::Ident(name) => {
            if let Some(ty) = type_map.get(name.as_str()) {
                e.ty = ty.clone();
            }
        }
        HirExprKind::BinOp { left, right, op } => {
            **left = retype_expr(left, type_map);
            **right = retype_expr(right, type_map);
            // Propagate: infer result type from children
            e.ty = infer_binop_type(&left.ty, &right.ty, *op);
        }
        HirExprKind::UnaryOp { operand, op: _ } => {
            **operand = retype_expr(operand, type_map);
            // Neg preserves type, Not → Bool
            e.ty = match &operand.ty {
                Type::Int => Type::Int,
                Type::Float => Type::Float,
                _ => e.ty.clone(),
            };
        }
        HirExprKind::Call { func, args } => {
            **func = retype_expr(func, type_map);
            for arg in args.iter_mut() {
                *arg = retype_expr(arg, type_map);
            }
            // Don't change the call's result type — it depends on the callee
        }
        HirExprKind::Index { target, index } => {
            **target = retype_expr(target, type_map);
            **index = retype_expr(index, type_map);
        }
        HirExprKind::Member { target, .. } => {
            **target = retype_expr(target, type_map);
        }
        HirExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            **cond = retype_expr(cond, type_map);
            if let Some(te) = then_expr {
                **te = retype_expr(te, type_map);
            }
            if let Some(ee) = else_expr {
                **ee = retype_expr(ee, type_map);
            }
        }
        HirExprKind::Block { expr: Some(e), .. } => {
            **e = retype_expr(e, type_map);
        }
        HirExprKind::Array(elems) | HirExprKind::Tuple(elems) => {
            for elem in elems.iter_mut() {
                *elem = retype_expr(elem, type_map);
            }
        }
        HirExprKind::Length(inner) => {
            **inner = retype_expr(inner, type_map);
        }
        // Literals and other nodes keep their types
        _ => {}
    }
    e
}

/// Infer the result type of a binary operation given the types of both operands.
/// Delegates to the canonical `tok_types::infer_binop_type` via `HirBinOp::to_parser_op()`.
pub(crate) fn infer_binop_type(left: &Type, right: &Type, op: HirBinOp) -> Type {
    tok_types::infer_binop_type(&op.to_parser_op(), left, right)
}
