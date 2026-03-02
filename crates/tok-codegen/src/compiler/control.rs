// ─── If/else & Loops ──────────────────────────────────────────────────

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::Variable;

use tok_hir::hir::*;
use tok_types::Type;

use super::{
    cl_type, cl_type_or_i64, coerce_if_branch, coerce_value, compile_body, compile_expr,
    from_tokvalue, to_bool, to_tokvalue, zero_value, FuncCtx,
};

// ─── If/else ──────────────────────────────────────────────────────────

pub(crate) fn compile_if(
    ctx: &mut FuncCtx,
    cond: &HirExpr,
    then_body: &[HirStmt],
    then_expr: &Option<Box<HirExpr>>,
    else_body: &[HirStmt],
    else_expr: &Option<Box<HirExpr>>,
    result_ty: &Type,
) -> Option<Value> {
    let cond_val = compile_expr(ctx, cond).expect("codegen: condition expr produced no value");
    let cond_bool = to_bool(ctx, cond_val, &cond.ty);

    let then_block = ctx.builder.create_block();
    let else_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();

    // If either branch is Any but result_ty is concrete, we must use Any semantics
    // internally because the Any branch might hold a different runtime type.
    let then_ty = then_expr.as_ref().map(|e| &e.ty);
    let else_ty = else_expr.as_ref().map(|e| &e.ty);
    let any_branch = then_ty.is_some_and(|t| matches!(t, Type::Any))
        || else_ty.is_some_and(|t| matches!(t, Type::Any));
    let needs_any_upgrade = any_branch && !matches!(result_ty, Type::Any | Type::Nil | Type::Never);
    let merge_ty = if needs_any_upgrade {
        &Type::Any
    } else {
        result_ty
    };

    // Does this if produce a value?
    let has_value = cl_type(merge_ty).is_some() && (then_expr.is_some() || else_expr.is_some());
    let result_cl_type = cl_type_or_i64(merge_ty);
    if has_value {
        ctx.builder.append_block_param(merge_block, result_cl_type);
    }

    ctx.builder
        .ins()
        .brif(cond_bool, then_block, &[], else_block, &[]);

    // Then branch
    ctx.builder.switch_to_block(then_block);
    ctx.builder.seal_block(then_block);
    ctx.block_terminated = false;
    compile_body(ctx, then_body, &Type::Nil);
    let then_val = if !ctx.block_terminated {
        if let Some(te) = then_expr {
            compile_expr(ctx, te)
        } else {
            None
        }
    } else {
        None
    };
    let then_terminated = ctx.block_terminated;
    if !then_terminated {
        if has_value {
            let v = then_val.unwrap_or_else(|| zero_value(&mut ctx.builder, result_cl_type));
            let then_expr_ty = then_expr.as_ref().map(|e| &e.ty);
            let v = coerce_if_branch(ctx, v, then_expr_ty, merge_ty, result_cl_type);
            ctx.builder.ins().jump(merge_block, &[v]);
        } else {
            ctx.builder.ins().jump(merge_block, &[]);
        }
    } else {
        // Block was terminated (return/break) — fill the dead block with a trap
        // so Cranelift doesn't complain about unfilled blocks
        ctx.builder
            .ins()
            .trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
    }

    // Else branch
    ctx.builder.switch_to_block(else_block);
    ctx.builder.seal_block(else_block);
    ctx.block_terminated = false;
    compile_body(ctx, else_body, &Type::Nil);
    let else_val = if !ctx.block_terminated {
        if let Some(ee) = else_expr {
            compile_expr(ctx, ee)
        } else {
            None
        }
    } else {
        None
    };
    let else_terminated = ctx.block_terminated;
    if !else_terminated {
        if has_value {
            let v = else_val.unwrap_or_else(|| zero_value(&mut ctx.builder, result_cl_type));
            let else_expr_ty = else_expr.as_ref().map(|e| &e.ty);
            let v = coerce_if_branch(ctx, v, else_expr_ty, merge_ty, result_cl_type);
            ctx.builder.ins().jump(merge_block, &[v]);
        } else {
            ctx.builder.ins().jump(merge_block, &[]);
        }
    } else {
        ctx.builder
            .ins()
            .trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
    }

    // If both branches terminated (return/break), the merge block is unreachable
    // but we still need to switch to it for subsequent code
    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
    ctx.block_terminated = then_terminated && else_terminated;

    if has_value {
        let merge_val = ctx.builder.block_params(merge_block)[0];
        // If we upgraded to Any internally, coerce back to the original result type
        if needs_any_upgrade {
            Some(coerce_value(ctx, merge_val, &Type::Any, result_ty))
        } else {
            Some(merge_val)
        }
    } else {
        None
    }
}

// ─── Loops ────────────────────────────────────────────────────────────

/// Check if a ForRange loop body is safe to unroll.
/// Criteria: no break/continue, no function calls, no nested loops, no returns.
/// Emit a loop variable increment (+1 or -1), condition check, and conditional branch.
pub(crate) fn emit_loop_increment(
    ctx: &mut FuncCtx,
    loop_var: Variable,
    ascending: bool,
    cc: cranelift_codegen::ir::condcodes::IntCC,
    limit: Value,
    continue_block: cranelift_codegen::ir::Block,
    exit_block: cranelift_codegen::ir::Block,
) {
    let current = ctx.builder.use_var(loop_var);
    let one = ctx.builder.ins().iconst(types::I64, 1);
    let next = if ascending {
        ctx.builder.ins().iadd(current, one)
    } else {
        ctx.builder.ins().isub(current, one)
    };
    ctx.builder.def_var(loop_var, next);
    let cond = ctx.builder.ins().icmp(cc, next, limit);
    ctx.builder
        .ins()
        .brif(cond, continue_block, &[], exit_block, &[]);
}

pub(crate) fn can_unroll_loop(body: &[HirStmt]) -> bool {
    for stmt in body {
        if !stmt_safe_to_unroll(stmt) {
            return false;
        }
    }
    true
}

pub(crate) fn stmt_safe_to_unroll(stmt: &HirStmt) -> bool {
    match stmt {
        HirStmt::Break | HirStmt::Continue | HirStmt::Return(_) => false,
        HirStmt::Import(_) => false,
        HirStmt::Assign { value, .. } => expr_safe_to_unroll(value),
        HirStmt::IndexAssign {
            target,
            index,
            value,
        } => {
            expr_safe_to_unroll(target) && expr_safe_to_unroll(index) && expr_safe_to_unroll(value)
        }
        HirStmt::MemberAssign { target, value, .. } => {
            expr_safe_to_unroll(target) && expr_safe_to_unroll(value)
        }
        HirStmt::Expr(e) => expr_safe_to_unroll(e),
        HirStmt::FuncDecl { .. } => false,
    }
}

pub(crate) fn expr_safe_to_unroll(expr: &HirExpr) -> bool {
    use HirExprKind::*;
    match &expr.kind {
        Int(_) | Float(_) | Str(_) | Bool(_) | Nil | Ident(_) => true,
        BinOp { left, right, .. } => expr_safe_to_unroll(left) && expr_safe_to_unroll(right),
        UnaryOp { operand, .. } => expr_safe_to_unroll(operand),
        // No function calls, no loops, no complex expressions
        Call { .. } | RuntimeCall { .. } | Lambda { .. } | Loop { .. } => false,
        If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            expr_safe_to_unroll(cond)
                && then_body.iter().all(stmt_safe_to_unroll)
                && then_expr.as_ref().is_none_or(|e| expr_safe_to_unroll(e))
                && else_body.iter().all(stmt_safe_to_unroll)
                && else_expr.as_ref().is_none_or(|e| expr_safe_to_unroll(e))
        }
        Index { target, index } => expr_safe_to_unroll(target) && expr_safe_to_unroll(index),
        Member { target, .. } => expr_safe_to_unroll(target),
        Array(elems) => elems.iter().all(expr_safe_to_unroll),
        Map(entries) => entries.iter().all(|(_, e)| expr_safe_to_unroll(e)),
        Tuple(elems) => elems.iter().all(expr_safe_to_unroll),
        Block { stmts, expr } => {
            stmts.iter().all(stmt_safe_to_unroll)
                && expr.as_ref().is_none_or(|e| expr_safe_to_unroll(e))
        }
        Length(e) => expr_safe_to_unroll(e),
        Range { start, end, .. } => expr_safe_to_unroll(start) && expr_safe_to_unroll(end),
        Go(_) | Receive(_) | Send { .. } | Select(_) => false,
    }
}

const UNROLL_FACTOR: i64 = 4;

pub(crate) fn compile_loop(ctx: &mut FuncCtx, kind: &HirLoopKind, body: &[HirStmt]) {
    match kind {
        HirLoopKind::While(cond) => {
            let header_block = ctx.builder.create_block();
            let body_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            ctx.builder.ins().jump(header_block, &[]);
            ctx.builder.switch_to_block(header_block);

            let cond_val =
                compile_expr(ctx, cond).expect("codegen: condition expr produced no value");
            let cond_bool = to_bool(ctx, cond_val, &cond.ty);
            ctx.builder
                .ins()
                .brif(cond_bool, body_block, &[], exit_block, &[]);

            ctx.builder.switch_to_block(body_block);
            ctx.builder.seal_block(body_block);

            ctx.loop_stack.push((header_block, exit_block));
            compile_body(ctx, body, &Type::Nil);
            ctx.loop_stack.pop();

            if !ctx.block_terminated {
                ctx.builder.ins().jump(header_block, &[]);
            }

            ctx.builder.seal_block(header_block);
            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
            ctx.block_terminated = false;
        }

        HirLoopKind::ForRange {
            var,
            start,
            end,
            inclusive,
        } => {
            let start_val =
                compile_expr(ctx, start).expect("codegen: start expr produced no value");
            let end_val = compile_expr(ctx, end).expect("codegen: end expr produced no value");
            // ForRange loop counter must be native i64 — unwrap/widen as needed
            let start_val = {
                let actual = ctx.builder.func.dfg.value_type(start_val);
                if actual == types::I8 {
                    ctx.builder.ins().uextend(types::I64, start_val)
                } else if matches!(&start.ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                    coerce_value(ctx, start_val, &start.ty, &Type::Int)
                } else {
                    start_val
                }
            };
            let end_val = {
                let actual = ctx.builder.func.dfg.value_type(end_val);
                if actual == types::I8 {
                    ctx.builder.ins().uextend(types::I64, end_val)
                } else if matches!(&end.ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                    coerce_value(ctx, end_val, &end.ty, &Type::Int)
                } else {
                    end_val
                }
            };

            // Create loop variable
            let loop_var = ctx.new_var(types::I64);
            ctx.builder.def_var(loop_var, start_val);
            ctx.vars.insert(var.clone(), (loop_var, Type::Int));

            let cc_asc = if *inclusive {
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThanOrEqual
            } else {
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan
            };

            // Determine if range is statically ascending (for unrolling optimization)
            let static_ascending = match (&start.kind, &end.kind) {
                (HirExprKind::Int(s), HirExprKind::Int(e)) => *s < *e,
                _ => false,
            };

            // Try loop unrolling for simple bodies (ascending, non-inclusive ranges only)
            if !*inclusive && can_unroll_loop(body) && static_ascending {
                let cc = cc_asc;
                // Unrolled loop: main loop steps by UNROLL_FACTOR, remainder loop handles leftovers
                let unrolled_body_block = ctx.builder.create_block();
                let unrolled_inc_block = ctx.builder.create_block();
                let remainder_body_block = ctx.builder.create_block();
                let remainder_inc_block = ctx.builder.create_block();
                let exit_block = ctx.builder.create_block();

                // Compute unrolled_end = start + ((end - start) / UNROLL_FACTOR) * UNROLL_FACTOR
                // This is the limit for the main unrolled loop
                let range_size = ctx.builder.ins().isub(end_val, start_val);
                let factor = ctx.builder.ins().iconst(types::I64, UNROLL_FACTOR);
                let full_chunks = ctx.builder.ins().sdiv(range_size, factor);
                let unrolled_count = ctx.builder.ins().imul(full_chunks, factor);
                let unrolled_end = ctx.builder.ins().iadd(start_val, unrolled_count);

                // Guard: skip unrolled loop if start >= unrolled_end (fewer than UNROLL_FACTOR iterations)
                let guard1 = ctx.builder.ins().icmp(cc, start_val, unrolled_end);
                ctx.builder
                    .ins()
                    .brif(guard1, unrolled_body_block, &[], remainder_body_block, &[]);

                // === Unrolled main loop body ===
                ctx.builder.switch_to_block(unrolled_body_block);

                // Push loop stack for break/continue (though unrolled bodies shouldn't have them)
                ctx.loop_stack.push((unrolled_inc_block, exit_block));

                // Emit body UNROLL_FACTOR times with loop var offset
                for u in 0..UNROLL_FACTOR {
                    if u > 0 {
                        let current = ctx.builder.use_var(loop_var);
                        let offset = ctx.builder.ins().iconst(types::I64, 1);
                        let next = ctx.builder.ins().iadd(current, offset);
                        ctx.builder.def_var(loop_var, next);
                    }
                    compile_body(ctx, body, &Type::Nil);
                }

                ctx.loop_stack.pop();

                if !ctx.block_terminated {
                    ctx.builder.ins().jump(unrolled_inc_block, &[]);
                }

                // Unrolled increment: advance by 1 more (total UNROLL_FACTOR) and check
                ctx.builder.switch_to_block(unrolled_inc_block);
                ctx.builder.seal_block(unrolled_inc_block);
                emit_loop_increment(
                    ctx,
                    loop_var,
                    true,
                    cc,
                    unrolled_end,
                    unrolled_body_block,
                    remainder_body_block,
                );

                ctx.builder.seal_block(unrolled_body_block);

                // === Remainder loop (handles leftover iterations) ===
                ctx.builder.switch_to_block(remainder_body_block);

                // Check if there are any remaining iterations
                let rem_current = ctx.builder.use_var(loop_var);
                let rem_guard = ctx.builder.ins().icmp(cc, rem_current, end_val);
                let remainder_real_block = ctx.builder.create_block();
                ctx.builder
                    .ins()
                    .brif(rem_guard, remainder_real_block, &[], exit_block, &[]);

                ctx.builder.seal_block(remainder_body_block);

                // Remainder body
                ctx.builder.switch_to_block(remainder_real_block);

                ctx.loop_stack.push((remainder_inc_block, exit_block));
                compile_body(ctx, body, &Type::Nil);
                ctx.loop_stack.pop();

                if !ctx.block_terminated {
                    ctx.builder.ins().jump(remainder_inc_block, &[]);
                }

                // Remainder increment
                ctx.builder.switch_to_block(remainder_inc_block);
                ctx.builder.seal_block(remainder_inc_block);
                emit_loop_increment(
                    ctx,
                    loop_var,
                    true,
                    cc,
                    end_val,
                    remainder_real_block,
                    exit_block,
                );

                ctx.builder.seal_block(remainder_real_block);
                ctx.builder.switch_to_block(exit_block);
                ctx.builder.seal_block(exit_block);
                ctx.block_terminated = false;
            } else {
                // Standard rotated loop with runtime direction support.
                // Handles both ascending (0..5) and descending (5..0) ranges.
                let cc_desc = if *inclusive {
                    cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThanOrEqual
                } else {
                    cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThan
                };

                let body_block = ctx.builder.create_block();
                let dispatch_block = ctx.builder.create_block();
                let asc_inc_block = ctx.builder.create_block();
                let desc_inc_block = ctx.builder.create_block();
                let exit_block = ctx.builder.create_block();

                // Determine direction at runtime
                let is_ascending = ctx.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                    start_val,
                    end_val,
                );

                // Guard: check if range has any iterations in either direction
                let asc_guard = ctx.builder.ins().icmp(cc_asc, start_val, end_val);
                let desc_guard = ctx.builder.ins().icmp(cc_desc, start_val, end_val);
                // Enter loop if ascending guard OR descending guard passes
                let enter_loop = ctx.builder.ins().bor(asc_guard, desc_guard);
                ctx.builder
                    .ins()
                    .brif(enter_loop, body_block, &[], exit_block, &[]);

                // Body block
                ctx.builder.switch_to_block(body_block);

                // continue jumps to dispatch_block which routes to correct inc
                ctx.loop_stack.push((dispatch_block, exit_block));
                compile_body(ctx, body, &Type::Nil);
                ctx.loop_stack.pop();

                if !ctx.block_terminated {
                    ctx.builder.ins().jump(dispatch_block, &[]);
                }

                // Dispatch block: route to ascending or descending increment
                ctx.builder.switch_to_block(dispatch_block);
                ctx.builder.seal_block(dispatch_block);
                ctx.builder
                    .ins()
                    .brif(is_ascending, asc_inc_block, &[], desc_inc_block, &[]);

                // Ascending increment: i += 1, check i < end
                ctx.builder.switch_to_block(asc_inc_block);
                ctx.builder.seal_block(asc_inc_block);
                emit_loop_increment(ctx, loop_var, true, cc_asc, end_val, body_block, exit_block);

                // Descending increment: i -= 1, check i > end
                ctx.builder.switch_to_block(desc_inc_block);
                ctx.builder.seal_block(desc_inc_block);
                emit_loop_increment(
                    ctx, loop_var, false, cc_desc, end_val, body_block, exit_block,
                );

                ctx.builder.seal_block(body_block);
                ctx.builder.switch_to_block(exit_block);
                ctx.builder.seal_block(exit_block);
                ctx.block_terminated = false;
            }
        }

        HirLoopKind::ForEach { var, iter } => {
            let iter_val =
                compile_expr(ctx, iter).expect("codegen: iterator expr produced no value");

            // For Any-typed iterables, extract the actual pointer and use runtime dispatch
            let is_any_iter = matches!(&iter.ty, Type::Any | Type::Optional(_) | Type::Result(_));

            // Get length
            let len_val = if is_any_iter {
                let (tag, data) = to_tokvalue(ctx, iter_val, &iter.ty);
                let len_ref = ctx.get_runtime_func_ref("tok_value_len");
                let len_call = ctx.builder.ins().call(len_ref, &[tag, data]);
                ctx.builder.inst_results(len_call)[0]
            } else {
                let len_func = match &iter.ty {
                    Type::Array(_) => "tok_array_len",
                    Type::Str => "tok_string_len",
                    _ => "tok_array_len",
                };
                let len_ref = ctx.get_runtime_func_ref(len_func);
                let len_call = ctx.builder.ins().call(len_ref, &[iter_val]);
                ctx.builder.inst_results(len_call)[0]
            };

            // Index variable
            let idx_var = ctx.new_var(types::I64);
            let zero = ctx.builder.ins().iconst(types::I64, 0);
            ctx.builder.def_var(idx_var, zero);

            // Element variable — for Any iterables, elements are Any too
            let elem_type = match &iter.ty {
                Type::Array(inner) => inner.as_ref().clone(),
                Type::Str => Type::Str,
                _ => Type::Any,
            };
            let ct = cl_type_or_i64(&elem_type);
            let elem_var = ctx.new_var(ct);
            let elem_zero = zero_value(&mut ctx.builder, ct);
            ctx.builder.def_var(elem_var, elem_zero);
            ctx.vars.insert(var.clone(), (elem_var, elem_type.clone()));

            let header_block = ctx.builder.create_block();
            let body_block = ctx.builder.create_block();
            let inc_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            ctx.builder.ins().jump(header_block, &[]);
            ctx.builder.switch_to_block(header_block);

            let current_idx = ctx.builder.use_var(idx_var);
            let cond = ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                current_idx,
                len_val,
            );
            ctx.builder
                .ins()
                .brif(cond, body_block, &[], exit_block, &[]);

            ctx.builder.switch_to_block(body_block);
            ctx.builder.seal_block(body_block);

            // Get element
            let current_idx = ctx.builder.use_var(idx_var);
            if is_any_iter {
                // For Any, use tok_value_index which dispatches by tag
                let (t_tag, t_data) = to_tokvalue(ctx, iter_val, &iter.ty);
                let idx_ref = ctx.get_runtime_func_ref("tok_value_index");
                let idx_call = ctx
                    .builder
                    .ins()
                    .call(idx_ref, &[t_tag, t_data, current_idx]);
                let results = ctx.builder.inst_results(idx_call);
                let elem = from_tokvalue(ctx, results[0], results[1], &elem_type);
                ctx.builder.def_var(elem_var, elem);
            } else {
                match &iter.ty {
                    Type::Array(_) => {
                        let get_ref = ctx.get_runtime_func_ref("tok_array_get");
                        let get_call = ctx.builder.ins().call(get_ref, &[iter_val, current_idx]);
                        let results = ctx.builder.inst_results(get_call);
                        let elem = from_tokvalue(ctx, results[0], results[1], &elem_type);
                        ctx.builder.def_var(elem_var, elem);
                    }
                    Type::Str => {
                        let get_ref = ctx.get_runtime_func_ref("tok_string_index");
                        let get_call = ctx.builder.ins().call(get_ref, &[iter_val, current_idx]);
                        let elem = ctx.builder.inst_results(get_call)[0];
                        ctx.builder.def_var(elem_var, elem);
                    }
                    _ => {
                        let get_ref = ctx.get_runtime_func_ref("tok_array_get");
                        let get_call = ctx.builder.ins().call(get_ref, &[iter_val, current_idx]);
                        let results = ctx.builder.inst_results(get_call);
                        let elem = from_tokvalue(ctx, results[0], results[1], &elem_type);
                        ctx.builder.def_var(elem_var, elem);
                    }
                }
            }

            ctx.loop_stack.push((inc_block, exit_block));
            compile_body(ctx, body, &Type::Nil);
            ctx.loop_stack.pop();

            ctx.builder.ins().jump(inc_block, &[]);

            ctx.builder.switch_to_block(inc_block);
            ctx.builder.seal_block(inc_block);
            let current_idx = ctx.builder.use_var(idx_var);
            let one = ctx.builder.ins().iconst(types::I64, 1);
            let next_idx = ctx.builder.ins().iadd(current_idx, one);
            ctx.builder.def_var(idx_var, next_idx);
            ctx.builder.ins().jump(header_block, &[]);

            ctx.builder.seal_block(header_block);
            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
        }

        HirLoopKind::ForEachIndexed {
            idx_var: idx_name,
            val_var: val_name,
            iter,
        } => {
            // Indexed foreach: ~(i v:collection){...}
            // For arrays: i = integer index, v = element
            // For maps: i = string key, v = value
            let iter_val =
                compile_expr(ctx, iter).expect("codegen: iterator expr produced no value");
            let is_map = matches!(&iter.ty, Type::Map(_));

            // Get length and (for maps) extract keys/vals arrays
            let (len_val, keys_arr, vals_arr) = if is_map {
                let len_ref = ctx.get_runtime_func_ref("tok_map_len");
                let len_call = ctx.builder.ins().call(len_ref, &[iter_val]);
                let len_val = ctx.builder.inst_results(len_call)[0];

                let keys_ref = ctx.get_runtime_func_ref("tok_map_keys");
                let keys_call = ctx.builder.ins().call(keys_ref, &[iter_val]);
                let keys_arr = ctx.builder.inst_results(keys_call)[0];

                let vals_ref = ctx.get_runtime_func_ref("tok_map_vals");
                let vals_call = ctx.builder.ins().call(vals_ref, &[iter_val]);
                let vals_arr = ctx.builder.inst_results(vals_call)[0];

                (len_val, Some(keys_arr), Some(vals_arr))
            } else {
                let len_ref = ctx.get_runtime_func_ref("tok_array_len");
                let len_call = ctx.builder.ins().call(len_ref, &[iter_val]);
                let len_val = ctx.builder.inst_results(len_call)[0];
                (len_val, None, None)
            };

            // Internal integer loop counter (always i64)
            let int_idx_var = ctx.new_var(types::I64);
            let zero = ctx.builder.ins().iconst(types::I64, 0);
            ctx.builder.def_var(int_idx_var, zero);

            // User-visible index/key variable
            let idx_type = if is_map { Type::Str } else { Type::Int };
            let idx_ct = cl_type_or_i64(&idx_type);
            let idx_var = ctx.new_var(idx_ct);
            let idx_zero = zero_value(&mut ctx.builder, idx_ct);
            ctx.builder.def_var(idx_var, idx_zero);
            ctx.vars.insert(idx_name.clone(), (idx_var, idx_type));

            // Value variable
            let elem_type = match &iter.ty {
                Type::Array(inner) => inner.as_ref().clone(),
                Type::Map(inner) => inner.as_ref().clone(),
                _ => Type::Any,
            };
            let ct = cl_type_or_i64(&elem_type);
            let elem_var = ctx.new_var(ct);
            let elem_zero = zero_value(&mut ctx.builder, ct);
            ctx.builder.def_var(elem_var, elem_zero);
            ctx.vars
                .insert(val_name.clone(), (elem_var, elem_type.clone()));

            let header_block = ctx.builder.create_block();
            let body_block = ctx.builder.create_block();
            let inc_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            ctx.builder.ins().jump(header_block, &[]);
            ctx.builder.switch_to_block(header_block);

            let current_idx = ctx.builder.use_var(int_idx_var);
            let cond = ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                current_idx,
                len_val,
            );
            ctx.builder
                .ins()
                .brif(cond, body_block, &[], exit_block, &[]);

            ctx.builder.switch_to_block(body_block);
            ctx.builder.seal_block(body_block);

            let current_idx = ctx.builder.use_var(int_idx_var);

            if is_map {
                // Map iteration: fetch key from keys array, value from vals array
                let get_ref = ctx.get_runtime_func_ref("tok_array_get");

                let get_key_call = ctx.builder.ins().call(
                    get_ref,
                    &[
                        keys_arr.expect("codegen: map iteration missing keys array"),
                        current_idx,
                    ],
                );
                let key_results = ctx.builder.inst_results(get_key_call);
                let key = from_tokvalue(ctx, key_results[0], key_results[1], &Type::Str);
                ctx.builder.def_var(idx_var, key);

                let get_val_call = ctx.builder.ins().call(
                    get_ref,
                    &[
                        vals_arr.expect("codegen: map iteration missing vals array"),
                        current_idx,
                    ],
                );
                let val_results = ctx.builder.inst_results(get_val_call);
                let val = from_tokvalue(ctx, val_results[0], val_results[1], &elem_type);
                ctx.builder.def_var(elem_var, val);
            } else {
                // Array iteration: index is the integer, value from array
                ctx.builder.def_var(idx_var, current_idx);

                let get_ref = ctx.get_runtime_func_ref("tok_array_get");
                let get_call = ctx.builder.ins().call(get_ref, &[iter_val, current_idx]);
                let results = ctx.builder.inst_results(get_call);
                let elem = from_tokvalue(ctx, results[0], results[1], &elem_type);
                ctx.builder.def_var(elem_var, elem);
            }

            ctx.loop_stack.push((inc_block, exit_block));
            compile_body(ctx, body, &Type::Nil);
            ctx.loop_stack.pop();

            if !ctx.block_terminated {
                ctx.builder.ins().jump(inc_block, &[]);
            }

            ctx.builder.switch_to_block(inc_block);
            ctx.builder.seal_block(inc_block);
            let current_idx = ctx.builder.use_var(int_idx_var);
            let one = ctx.builder.ins().iconst(types::I64, 1);
            let next_idx = ctx.builder.ins().iadd(current_idx, one);
            ctx.builder.def_var(int_idx_var, next_idx);
            ctx.builder.ins().jump(header_block, &[]);

            ctx.builder.seal_block(header_block);
            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
        }

        HirLoopKind::Infinite => {
            let body_block = ctx.builder.create_block();
            let exit_block = ctx.builder.create_block();

            ctx.builder.ins().jump(body_block, &[]);
            ctx.builder.switch_to_block(body_block);

            ctx.loop_stack.push((body_block, exit_block));
            compile_body(ctx, body, &Type::Nil);
            ctx.loop_stack.pop();

            if !ctx.block_terminated {
                ctx.builder.ins().jump(body_block, &[]);
            }

            ctx.builder.seal_block(body_block);
            ctx.builder.switch_to_block(exit_block);
            ctx.builder.seal_block(exit_block);
            ctx.block_terminated = false;
        }
    }
}
