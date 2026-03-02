// ─── Inline closure calls & filter/reduce ──────────────────────────────

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{InstBuilder, MemFlags, Value};
use cranelift_frontend::Variable;

use std::collections::HashMap;

use tok_hir::hir::*;
use tok_types::Type;

use super::{
    alloc_tokvalue_on_stack, cl_type_or_i64, coerce_value, compile_body, compile_expr,
    from_tokvalue, load_captures_from_env, retype_body, to_bool, to_tokvalue,
    unwrap_return_stmts, zero_value, FuncCtx, KnownClosure, PendingLambda, TAG_ARRAY,
};

// ─── Inline closure calls ──────────────────────────────────────────────

/// Check if a known closure can be inlined at its call site.
/// Returns true if the lambda body is a single expression and all types are concrete.
pub(crate) fn can_inline_closure_call(pending: &PendingLambda, arg_types: &[Type], var_name: &str) -> bool {
    // All args must be concrete
    if !arg_types
        .iter()
        .all(|t| matches!(t, Type::Int | Type::Float | Type::Bool))
    {
        return false;
    }
    // Body must be exactly one statement: Expr(e) or Return(Some(e))
    if pending.body.len() != 1 {
        return false;
    }
    let body_expr = match &pending.body[0] {
        HirStmt::Expr(e) => e,
        HirStmt::Return(Some(e)) => e,
        _ => return false,
    };
    // All captures must be concrete
    if !pending
        .captures
        .iter()
        .all(|c| matches!(c.ty, Type::Int | Type::Float | Type::Bool))
    {
        return false;
    }
    // Don't inline self-recursive calls
    !contains_self_call(body_expr, var_name)
}

/// Check if an HIR statement contains a call to a function with the given name.
fn stmt_contains_self_call(stmt: &HirStmt, name: &str) -> bool {
    match stmt {
        HirStmt::Expr(e) => contains_self_call(e, name),
        HirStmt::Return(Some(e)) => contains_self_call(e, name),
        HirStmt::Assign { value, .. } => contains_self_call(value, name),
        _ => false,
    }
}

/// Check if an HIR expression contains a call to a function with the given name.
pub(crate) fn contains_self_call(expr: &HirExpr, name: &str) -> bool {
    match &expr.kind {
        HirExprKind::Call { func, args } => {
            if let HirExprKind::Ident(callee) = &func.kind {
                if callee == name {
                    return true;
                }
            }
            contains_self_call(func, name) || args.iter().any(|a| contains_self_call(a, name))
        }
        HirExprKind::BinOp { left, right, .. } => {
            contains_self_call(left, name) || contains_self_call(right, name)
        }
        HirExprKind::UnaryOp { operand, .. } => contains_self_call(operand, name),
        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            contains_self_call(cond, name)
                || then_body.iter().any(|s| stmt_contains_self_call(s, name))
                || then_expr
                    .as_ref()
                    .is_some_and(|e| contains_self_call(e, name))
                || else_body.iter().any(|s| stmt_contains_self_call(s, name))
                || else_expr
                    .as_ref()
                    .is_some_and(|e| contains_self_call(e, name))
        }
        HirExprKind::Index { target, index } => {
            contains_self_call(target, name) || contains_self_call(index, name)
        }
        HirExprKind::Member { target, .. } => contains_self_call(target, name),
        _ => false,
    }
}

/// Check if a function body ends with a self-tail-call (directly or in both branches of an if).
pub(crate) fn is_self_tail_recursive(body: &[HirStmt], name: &str) -> bool {
    let last = match body.last() {
        Some(s) => s,
        None => return false,
    };
    match last {
        HirStmt::Expr(e) => is_tail_call_expr(e, name),
        HirStmt::Return(Some(e)) => is_tail_call_expr(e, name),
        _ => false,
    }
}

/// Check if an expression is a tail call to `name` (direct call or if-then-else where at least
/// one branch is a tail call).
fn is_tail_call_expr(expr: &HirExpr, name: &str) -> bool {
    match &expr.kind {
        HirExprKind::Call { func, args: _ } => {
            if let HirExprKind::Ident(callee) = &func.kind {
                callee == name
            } else {
                false
            }
        }
        HirExprKind::If {
            then_body: _,
            then_expr,
            else_body: _,
            else_expr,
            ..
        } => {
            // At least one branch must be a tail call for TCO to be useful.
            // We only transform the branches that ARE tail calls; non-tail branches
            // return normally.
            let then_tail = then_expr
                .as_ref()
                .is_some_and(|e| is_tail_call_expr(e, name));
            let else_tail = else_expr
                .as_ref()
                .is_some_and(|e| is_tail_call_expr(e, name));
            then_tail || else_tail
        }
        _ => false,
    }
}

/// Inline a known lambda call at the call site.
/// Instead of emitting a function call, we compile the lambda body directly
/// into the caller's instruction stream.
pub(crate) fn compile_inline_closure_call(
    ctx: &mut FuncCtx,
    _name: &str,
    kc: &KnownClosure,
    args: &[HirExpr],
    arg_types: &[Type],
    result_ty: &Type,
) -> Option<Value> {
    // Get the pending lambda's body and metadata
    let params = ctx.compiler.pending_lambdas[kc.pending_idx].params.clone();
    let captures = ctx.compiler.pending_lambdas[kc.pending_idx]
        .captures
        .clone();
    let body = ctx.compiler.pending_lambdas[kc.pending_idx].body.clone();

    // Compile argument expressions before binding anything
    let mut arg_vals = Vec::new();
    for arg in args {
        let v = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        arg_vals.push(v);
    }

    // Save old bindings and bind lambda parameters to compiled arg values
    let mut old_param_bindings: Vec<(String, Option<(Variable, Type)>)> = Vec::new();
    for (i, param) in params.iter().enumerate() {
        old_param_bindings.push((param.name.clone(), ctx.vars.remove(&param.name)));
        let ct = cl_type_or_i64(&arg_types[i]);
        let var = ctx.new_var(ct);
        ctx.builder.def_var(var, arg_vals[i]);
        ctx.vars
            .insert(param.name.clone(), (var, arg_types[i].clone()));
    }

    // Handle captures: load from env_ptr to preserve snapshot semantics
    let mut old_capture_bindings: Vec<(String, Option<(Variable, Type)>)> = Vec::new();
    if !captures.is_empty() {
        let env_ptr = kc.env_ptr;
        for cap in captures.iter() {
            old_capture_bindings.push((cap.name.clone(), ctx.vars.remove(&cap.name)));
        }
        load_captures_from_env(ctx, &captures, env_ptr, true);
    }

    // Build type map and retype body for concrete types
    let mut type_map = HashMap::new();
    for (param, arg_ty) in params.iter().zip(arg_types.iter()) {
        type_map.insert(param.name.clone(), arg_ty.clone());
    }
    for cap in &captures {
        type_map.insert(cap.name.clone(), cap.ty.clone());
    }
    let retyped = retype_body(&body, &type_map);
    let retyped = unwrap_return_stmts(retyped);

    // Determine result type from the retyped body
    let body_result_ty = retyped
        .last()
        .and_then(|s| match s {
            HirStmt::Expr(e) => Some(e.ty.clone()),
            _ => None,
        })
        .unwrap_or(Type::Int);

    // Compile the retyped body inline
    let body_result = compile_body(ctx, &retyped, &body_result_ty);

    // Restore old bindings for parameters
    for (pname, old) in old_param_bindings {
        ctx.vars.remove(&pname);
        if let Some(old_val) = old {
            ctx.vars.insert(pname, old_val);
        }
    }
    // Restore old bindings for captures
    for (cname, old) in old_capture_bindings {
        ctx.vars.remove(&cname);
        if let Some(old_val) = old {
            ctx.vars.insert(cname, old_val);
        }
    }

    // Coerce result to caller's expected type
    if let Some(val) = body_result {
        if matches!(result_ty, Type::Any | Type::Optional(_) | Type::Result(_))
            && !matches!(body_result_ty, Type::Any)
        {
            let (tag, data) = to_tokvalue(ctx, val, &body_result_ty);
            Some(alloc_tokvalue_on_stack(ctx, tag, data))
        } else {
            Some(coerce_value(ctx, val, &body_result_ty, result_ty))
        }
    } else {
        None
    }
}

// ─── Inline filter/reduce ──────────────────────────────────────────────

/// Check if a filter/reduce lambda can be inlined at compile time.
pub(crate) fn can_inline_hof(lambda_expr: &HirExpr, arr_ty: &Type, expected_params: usize) -> bool {
    if let HirExprKind::Lambda { params, .. } = &lambda_expr.kind {
        if params.len() != expected_params {
            return false;
        }
        // Array element type must be concrete (not Any) for native-type inlining to help
        match arr_ty {
            Type::Array(inner) => !matches!(inner.as_ref(), Type::Any),
            _ => false,
        }
    } else {
        false
    }
}

/// Compile `arr?>\(x)=pred` as an inline loop instead of a runtime call.
pub(crate) fn compile_inline_filter(
    ctx: &mut FuncCtx,
    arr_expr: &HirExpr,
    lambda_expr: &HirExpr,
    result_ty: &Type,
) -> Option<Value> {
    let HirExprKind::Lambda { params, body, .. } = &lambda_expr.kind else {
        unreachable!()
    };
    let elem_type = match &arr_expr.ty {
        Type::Array(inner) => inner.as_ref().clone(),
        _ => Type::Any,
    };
    let param_name = &params[0].name;

    // Compile source array
    let arr_raw = compile_expr(ctx, arr_expr).expect("codegen: array expr produced no value");
    let arr = if matches!(
        &arr_expr.ty,
        Type::Any | Type::Optional(_) | Type::Result(_)
    ) {
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), arr_raw, 8)
    } else {
        arr_raw
    };

    // Allocate result array
    let alloc_ref = ctx.get_runtime_func_ref("tok_array_alloc");
    let alloc_call = ctx.builder.ins().call(alloc_ref, &[]);
    let result_arr = ctx.builder.inst_results(alloc_call)[0];

    // Get source length
    let len_ref = ctx.get_runtime_func_ref("tok_array_len");
    let len_call = ctx.builder.ins().call(len_ref, &[arr]);
    let len_val = ctx.builder.inst_results(len_call)[0];

    // Loop index
    let idx_var = ctx.new_var(types::I64);
    let zero = ctx.builder.ins().iconst(types::I64, 0);
    ctx.builder.def_var(idx_var, zero);

    // Element variable bound to lambda param
    let ct = cl_type_or_i64(&elem_type);
    let elem_var = ctx.new_var(ct);
    let elem_zero = zero_value(&mut ctx.builder, ct);
    ctx.builder.def_var(elem_var, elem_zero);

    // Save old binding and insert lambda param
    let old_binding = ctx.vars.remove(param_name);
    ctx.vars
        .insert(param_name.clone(), (elem_var, elem_type.clone()));

    // Loop blocks
    let header_block = ctx.builder.create_block();
    let body_block = ctx.builder.create_block();
    let push_block = ctx.builder.create_block();
    let inc_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();

    ctx.builder.ins().jump(header_block, &[]);
    ctx.builder.switch_to_block(header_block);

    // Condition: i < len
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

    // Get element as (tag, data)
    let current_idx = ctx.builder.use_var(idx_var);
    let get_ref = ctx.get_runtime_func_ref("tok_array_get");
    let get_call = ctx.builder.ins().call(get_ref, &[arr, current_idx]);
    let get_results = ctx.builder.inst_results(get_call);
    let elem_tag = get_results[0];
    let elem_data = get_results[1];

    // Extract native value for the lambda body
    let elem_native = from_tokvalue(ctx, elem_tag, elem_data, &elem_type);
    ctx.builder.def_var(elem_var, elem_native);

    // Retype and compile lambda body inline
    let mut type_map = HashMap::new();
    type_map.insert(param_name.clone(), elem_type.clone());
    let retyped = retype_body(body, &type_map);
    // Unwrap trailing Return(Some(expr)) → Expr(expr) since we're inlining
    let retyped = unwrap_return_stmts(retyped);
    let pred_result = compile_body(ctx, &retyped, &Type::Bool);

    if let Some(pred_val) = pred_result {
        if !ctx.block_terminated {
            // Determine predicate result type
            let pred_ty = retyped
                .last()
                .and_then(|s| match s {
                    HirStmt::Expr(e) => Some(e.ty.clone()),
                    _ => None,
                })
                .unwrap_or(Type::Bool);
            let bool_val = to_bool(ctx, pred_val, &pred_ty);

            ctx.builder
                .ins()
                .brif(bool_val, push_block, &[], inc_block, &[]);
        }

        // Push block: add element to result array
        ctx.builder.switch_to_block(push_block);
        ctx.builder.seal_block(push_block);
        let push_ref = ctx.get_runtime_func_ref("tok_array_push");
        ctx.builder
            .ins()
            .call(push_ref, &[result_arr, elem_tag, elem_data]);
        ctx.builder.ins().jump(inc_block, &[]);
    } else if !ctx.block_terminated {
        ctx.builder.ins().jump(inc_block, &[]);
    }

    // Increment
    ctx.block_terminated = false;
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
    ctx.block_terminated = false;

    // Restore old binding
    ctx.vars.remove(param_name);
    if let Some(old) = old_binding {
        ctx.vars.insert(param_name.clone(), old);
    }

    // If caller expects Any, wrap result
    if matches!(result_ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
        let tag = ctx.builder.ins().iconst(types::I64, TAG_ARRAY);
        return Some(alloc_tokvalue_on_stack(ctx, tag, result_arr));
    }
    Some(result_arr)
}

/// Compile `arr/>init \(acc x)=body` as an inline loop instead of a runtime call.
pub(crate) fn compile_inline_reduce(
    ctx: &mut FuncCtx,
    arr_expr: &HirExpr,
    init_expr: &HirExpr,
    lambda_expr: &HirExpr,
    result_ty: &Type,
) -> Option<Value> {
    let HirExprKind::Lambda { params, body, .. } = &lambda_expr.kind else {
        unreachable!()
    };
    let elem_type = match &arr_expr.ty {
        Type::Array(inner) => inner.as_ref().clone(),
        _ => Type::Any,
    };
    let acc_name = &params[0].name;
    let elem_name = &params[1].name;

    // Compile source array
    let arr_raw = compile_expr(ctx, arr_expr).expect("codegen: array expr produced no value");
    let arr = if matches!(
        &arr_expr.ty,
        Type::Any | Type::Optional(_) | Type::Result(_)
    ) {
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), arr_raw, 8)
    } else {
        arr_raw
    };

    // Get length
    let len_ref = ctx.get_runtime_func_ref("tok_array_len");
    let len_call = ctx.builder.ins().call(len_ref, &[arr]);
    let len_val = ctx.builder.inst_results(len_call)[0];

    // Determine accumulator type from init expression
    let acc_type = if matches!(&init_expr.kind, HirExprKind::Nil) {
        // No explicit init — acc type is same as element type
        elem_type.clone()
    } else {
        init_expr.ty.clone()
    };
    let acc_ct = cl_type_or_i64(&acc_type);

    // Compile init value and determine start index
    let (init_val, start_idx) = if matches!(&init_expr.kind, HirExprKind::Nil) {
        // No init: use first element, start from 1
        let zero_idx = ctx.builder.ins().iconst(types::I64, 0);
        let get_ref = ctx.get_runtime_func_ref("tok_array_get");
        let get_call = ctx.builder.ins().call(get_ref, &[arr, zero_idx]);
        let results = ctx.builder.inst_results(get_call);
        let first = from_tokvalue(ctx, results[0], results[1], &elem_type);
        let one = ctx.builder.ins().iconst(types::I64, 1);
        (first, one)
    } else {
        let iv = compile_expr(ctx, init_expr).expect("codegen: init expr produced no value");
        let zero = ctx.builder.ins().iconst(types::I64, 0);
        (iv, zero)
    };

    // Accumulator variable
    let acc_var = ctx.new_var(acc_ct);
    ctx.builder.def_var(acc_var, init_val);

    // Loop index
    let idx_var = ctx.new_var(types::I64);
    ctx.builder.def_var(idx_var, start_idx);

    // Element variable
    let elem_ct = cl_type_or_i64(&elem_type);
    let elem_var = ctx.new_var(elem_ct);
    let elem_zero = zero_value(&mut ctx.builder, elem_ct);
    ctx.builder.def_var(elem_var, elem_zero);

    // Bind lambda params
    let old_acc_binding = ctx.vars.remove(acc_name);
    let old_elem_binding = ctx.vars.remove(elem_name);
    ctx.vars
        .insert(acc_name.clone(), (acc_var, acc_type.clone()));
    ctx.vars
        .insert(elem_name.clone(), (elem_var, elem_type.clone()));

    // Loop blocks
    let header_block = ctx.builder.create_block();
    let body_block = ctx.builder.create_block();
    let inc_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();

    ctx.builder.ins().jump(header_block, &[]);
    ctx.builder.switch_to_block(header_block);

    // Condition: i < len
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
    let get_ref = ctx.get_runtime_func_ref("tok_array_get");
    let get_call = ctx.builder.ins().call(get_ref, &[arr, current_idx]);
    let get_results = ctx.builder.inst_results(get_call);
    let elem_native = from_tokvalue(ctx, get_results[0], get_results[1], &elem_type);
    ctx.builder.def_var(elem_var, elem_native);

    // Retype and compile lambda body inline
    let mut type_map = HashMap::new();
    type_map.insert(acc_name.clone(), acc_type.clone());
    type_map.insert(elem_name.clone(), elem_type.clone());
    let retyped = retype_body(body, &type_map);
    // Unwrap trailing Return(Some(expr)) → Expr(expr) since we're inlining
    let retyped = unwrap_return_stmts(retyped);
    let body_result = compile_body(ctx, &retyped, &acc_type);

    if let Some(val) = body_result {
        // Determine body result type
        let body_ty = retyped
            .last()
            .and_then(|s| match s {
                HirStmt::Expr(e) => Some(e.ty.clone()),
                _ => None,
            })
            .unwrap_or(acc_type.clone());
        let coerced = coerce_value(ctx, val, &body_ty, &acc_type);
        ctx.builder.def_var(acc_var, coerced);
    }

    // Increment
    if !ctx.block_terminated {
        ctx.builder.ins().jump(inc_block, &[]);
    }
    ctx.block_terminated = false;

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
    ctx.block_terminated = false;

    // Restore bindings
    ctx.vars.remove(acc_name);
    ctx.vars.remove(elem_name);
    if let Some(old) = old_acc_binding {
        ctx.vars.insert(acc_name.clone(), old);
    }
    if let Some(old) = old_elem_binding {
        ctx.vars.insert(elem_name.clone(), old);
    }

    // Return accumulator
    let final_acc = ctx.builder.use_var(acc_var);
    if matches!(result_ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
        let (tag, data) = to_tokvalue(ctx, final_acc, &acc_type);
        Some(alloc_tokvalue_on_stack(ctx, tag, data))
    } else {
        Some(coerce_value(ctx, final_acc, &acc_type, result_ty))
    }
}

pub(crate) fn compile_print_call(ctx: &mut FuncCtx, args: &[HirExpr], newline: bool) -> Option<Value> {
    for (i, arg) in args.iter().enumerate() {
        let val = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        let use_newline = newline && i == args.len() - 1;
        let func_name = match &arg.ty {
            Type::Int => {
                if use_newline {
                    "tok_println_int"
                } else {
                    "tok_print_int"
                }
            }
            Type::Float => {
                if use_newline {
                    "tok_println_float"
                } else {
                    "tok_print_float"
                }
            }
            Type::Str => {
                if use_newline {
                    "tok_println_string"
                } else {
                    "tok_print_string"
                }
            }
            Type::Bool => {
                if use_newline {
                    "tok_println_bool"
                } else {
                    "tok_print_bool"
                }
            }
            _ => {
                // Pack as TokValue
                let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                let func_name = if use_newline {
                    "tok_println"
                } else {
                    "tok_print"
                };
                let func_ref = ctx.get_runtime_func_ref(func_name);
                ctx.builder.ins().call(func_ref, &[tag, data]);
                continue;
            }
        };
        let func_ref = ctx.get_runtime_func_ref(func_name);
        ctx.builder.ins().call(func_ref, &[val]);
    }
    // Return a dummy value so the caller knows this builtin was handled.
    // Returning None would cause the caller to fall through to the generic
    // closure-call path, which tries to look up "pl" as a variable and segfaults.
    Some(ctx.builder.ins().iconst(types::I64, 0))
}
