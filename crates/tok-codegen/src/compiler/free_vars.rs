// ─── Free variable analysis for closure captures ──────────────────────

use std::collections::HashSet;

use tok_hir::hir::*;

/// Maximum HIR nesting depth before aborting free-var collection.
/// Prevents stack overflow on adversarially nested input.
const MAX_FREE_VAR_DEPTH: usize = 1000;

/// Collect all free variables referenced in a lambda body that are not in `bound` (params + locals).
/// Returns the names of variables that need to be captured from the enclosing scope.
pub(crate) fn collect_free_vars(body: &[HirStmt], param_names: &HashSet<String>) -> HashSet<String> {
    let mut free = HashSet::new();
    let mut locals = param_names.clone();
    for stmt in body {
        collect_free_vars_stmt(stmt, &mut locals, &mut free, 0);
    }
    free
}

fn collect_free_vars_stmt(
    stmt: &HirStmt,
    locals: &mut HashSet<String>,
    free: &mut HashSet<String>,
    depth: usize,
) {
    if depth >= MAX_FREE_VAR_DEPTH {
        return;
    }
    match stmt {
        HirStmt::Assign { name, value, .. } => {
            // The RHS may reference free vars (before the local is defined)
            collect_free_vars_expr(value, locals, free, depth + 1);
            locals.insert(name.clone());
        }
        HirStmt::Expr(expr) => {
            collect_free_vars_expr(expr, locals, free, depth + 1);
        }
        HirStmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr {
                collect_free_vars_expr(expr, locals, free, depth + 1);
            }
        }
        HirStmt::FuncDecl { name, .. } => {
            // The function name becomes a local, its body is a separate scope
            locals.insert(name.clone());
        }
        HirStmt::IndexAssign {
            target,
            index,
            value,
            ..
        } => {
            collect_free_vars_expr(target, locals, free, depth + 1);
            collect_free_vars_expr(index, locals, free, depth + 1);
            collect_free_vars_expr(value, locals, free, depth + 1);
        }
        HirStmt::MemberAssign { target, value, .. } => {
            collect_free_vars_expr(target, locals, free, depth + 1);
            collect_free_vars_expr(value, locals, free, depth + 1);
        }
        HirStmt::Import(_) => {}
        HirStmt::Break | HirStmt::Continue => {}
    }
}

pub(crate) fn collect_free_vars_expr(
    expr: &HirExpr,
    locals: &HashSet<String>,
    free: &mut HashSet<String>,
    depth: usize,
) {
    if depth >= MAX_FREE_VAR_DEPTH {
        return;
    }
    let d = depth + 1;
    match &expr.kind {
        HirExprKind::Ident(name) => {
            if !locals.contains(name) {
                free.insert(name.clone());
            }
        }
        HirExprKind::Int(_) | HirExprKind::Float(_) | HirExprKind::Bool(_) | HirExprKind::Nil => {}
        HirExprKind::Str(_) => {}
        HirExprKind::Array(elems) => {
            for e in elems {
                collect_free_vars_expr(e, locals, free, d);
            }
        }
        HirExprKind::Map(entries) => {
            for (_k, v) in entries {
                collect_free_vars_expr(v, locals, free, d);
            }
        }
        HirExprKind::Tuple(elems) => {
            for e in elems {
                collect_free_vars_expr(e, locals, free, d);
            }
        }
        HirExprKind::BinOp { left, right, .. } => {
            collect_free_vars_expr(left, locals, free, d);
            collect_free_vars_expr(right, locals, free, d);
        }
        HirExprKind::UnaryOp { operand, .. } => {
            collect_free_vars_expr(operand, locals, free, d);
        }
        HirExprKind::Index { target, index } => {
            collect_free_vars_expr(target, locals, free, d);
            collect_free_vars_expr(index, locals, free, d);
        }
        HirExprKind::Member { target, .. } => {
            collect_free_vars_expr(target, locals, free, d);
        }
        HirExprKind::Call { func, args } => {
            collect_free_vars_expr(func, locals, free, d);
            for a in args {
                collect_free_vars_expr(a, locals, free, d);
            }
        }
        HirExprKind::RuntimeCall { args, .. } => {
            for a in args {
                collect_free_vars_expr(a, locals, free, d);
            }
        }
        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            collect_free_vars_expr(cond, locals, free, d);
            for s in then_body {
                collect_free_vars_stmt(s, &mut locals.clone(), free, d);
            }
            if let Some(e) = then_expr {
                collect_free_vars_expr(e, locals, free, d);
            }
            for s in else_body {
                collect_free_vars_stmt(s, &mut locals.clone(), free, d);
            }
            if let Some(e) = else_expr {
                collect_free_vars_expr(e, locals, free, d);
            }
        }
        HirExprKind::Loop { kind, body } => {
            // Collect free vars from the loop kind (range start/end, condition, iterator)
            let mut loop_locals = locals.clone();
            match kind.as_ref() {
                HirLoopKind::ForRange {
                    var, start, end, ..
                } => {
                    collect_free_vars_expr(start, locals, free, d);
                    collect_free_vars_expr(end, locals, free, d);
                    loop_locals.insert(var.clone());
                }
                HirLoopKind::ForEach { var, iter } => {
                    collect_free_vars_expr(iter, locals, free, d);
                    loop_locals.insert(var.clone());
                }
                HirLoopKind::ForEachIndexed {
                    idx_var,
                    val_var,
                    iter,
                } => {
                    collect_free_vars_expr(iter, locals, free, d);
                    loop_locals.insert(idx_var.clone());
                    loop_locals.insert(val_var.clone());
                }
                HirLoopKind::While(cond) => {
                    collect_free_vars_expr(cond, locals, free, d);
                }
                HirLoopKind::Infinite => {}
            }
            for s in body {
                collect_free_vars_stmt(s, &mut loop_locals, free, d);
            }
        }
        HirExprKind::Lambda { params, body, .. } => {
            // Nested lambda: its params are bound, but it may capture from our scope
            let mut inner_locals = locals.clone();
            for p in params {
                inner_locals.insert(p.name.clone());
            }
            for s in body {
                collect_free_vars_stmt(s, &mut inner_locals, free, d);
            }
        }
        HirExprKind::Length(inner) => {
            collect_free_vars_expr(inner, locals, free, d);
        }
        HirExprKind::Block { stmts, expr } => {
            let mut block_locals = locals.clone();
            for s in stmts {
                collect_free_vars_stmt(s, &mut block_locals, free, d);
            }
            if let Some(e) = expr {
                collect_free_vars_expr(e, &block_locals, free, d);
            }
        }
        HirExprKind::Range { start, end, .. } => {
            collect_free_vars_expr(start, locals, free, d);
            collect_free_vars_expr(end, locals, free, d);
        }
        HirExprKind::Go(inner) => {
            collect_free_vars_expr(inner, locals, free, d);
        }
        HirExprKind::Receive(inner) => {
            collect_free_vars_expr(inner, locals, free, d);
        }
        HirExprKind::Send { chan, value } => {
            collect_free_vars_expr(chan, locals, free, d);
            collect_free_vars_expr(value, locals, free, d);
        }
        HirExprKind::Select(arms) => {
            for arm in arms {
                match arm {
                    HirSelectArm::Recv { chan, body, .. } => {
                        collect_free_vars_expr(chan, locals, free, d);
                        for s in body {
                            collect_free_vars_stmt(s, &mut locals.clone(), free, d);
                        }
                    }
                    HirSelectArm::Send { chan, value, body } => {
                        collect_free_vars_expr(chan, locals, free, d);
                        collect_free_vars_expr(value, locals, free, d);
                        for s in body {
                            collect_free_vars_stmt(s, &mut locals.clone(), free, d);
                        }
                    }
                    HirSelectArm::Default(body) => {
                        for s in body {
                            collect_free_vars_stmt(s, &mut locals.clone(), free, d);
                        }
                    }
                }
            }
        }
    }
}
