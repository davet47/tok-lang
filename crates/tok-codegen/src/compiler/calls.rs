// ─── Function calls ───────────────────────────────────────────────────

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_frontend::Variable;
use cranelift_module::{FuncId, Linkage, Module};

use std::collections::HashMap;

use tok_hir::hir::*;
use tok_types::Type;

use super::{
    alloc_tokvalue_on_stack, can_inline_closure_call, cl_type_or_i64, coerce_value, compile_body,
    compile_expr, compile_expr_as_ptr, compile_inline_closure_call, compile_print_call,
    contains_self_call, from_tokvalue, from_tokvalue_raw_data, get_stdlib_func, retype_body,
    to_tokvalue, unwrap_any_ptr, unwrap_return_stmts, zero_value, FuncCtx, KnownClosure,
    PendingLambda, PTR, TAG_ARRAY, TAG_FLOAT, TAG_INT, TAG_STRING,
};

// ─── Builtin call helpers ─────────────────────────────────────────────

/// Compile a 1-arg builtin: compile arg → unwrap_any_ptr → call runtime → return ptr
fn compile_builtin_1_ptr(ctx: &mut FuncCtx, arg: &HirExpr, runtime_fn: &str) -> Option<Value> {
    let ptr = compile_expr_as_ptr(ctx, arg);
    let func_ref = ctx.get_runtime_func_ref(runtime_fn);
    let call = ctx.builder.ins().call(func_ref, &[ptr]);
    Some(ctx.builder.inst_results(call)[0])
}

/// Compile a 1-arg builtin that returns (tag, data): compile arg → unwrap → call → from_tokvalue
fn compile_builtin_1_tokvalue(
    ctx: &mut FuncCtx,
    arg: &HirExpr,
    runtime_fn: &str,
    result_ty: &Type,
) -> Option<Value> {
    let ptr = compile_expr_as_ptr(ctx, arg);
    let func_ref = ctx.get_runtime_func_ref(runtime_fn);
    let call = ctx.builder.ins().call(func_ref, &[ptr]);
    let results = ctx.builder.inst_results(call);
    Some(from_tokvalue(ctx, results[0], results[1], result_ty))
}

/// Compile a 2-arg builtin: compile both → unwrap → call → return ptr
fn compile_builtin_2_ptr(ctx: &mut FuncCtx, args: &[HirExpr], runtime_fn: &str) -> Option<Value> {
    let a = compile_expr_as_ptr(ctx, &args[0]);
    let b = compile_expr_as_ptr(ctx, &args[1]);
    let func_ref = ctx.get_runtime_func_ref(runtime_fn);
    let call = ctx.builder.ins().call(func_ref, &[a, b]);
    Some(ctx.builder.inst_results(call)[0])
}

pub(crate) fn compile_call(
    ctx: &mut FuncCtx,
    func_expr: &HirExpr,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    // ── Stdlib direct-call optimization ─────────────────────────
    // Detect m.func(args) where m is a known stdlib import.
    // Emit a direct call to the trampoline, bypassing map lookup + indirect dispatch.
    if let HirExprKind::Member { target, field } = &func_expr.kind {
        if let HirExprKind::Ident(var_name) = &target.kind {
            if let Some(module_name) = ctx.stdlib_imports.get(var_name.as_str()).cloned() {
                if let Some((trampoline, _arity)) = get_stdlib_func(&module_name, field) {
                    let func_ref = ctx.get_runtime_func_ref(trampoline);
                    let null_env = ctx.builder.ins().iconst(PTR, 0i64);
                    let mut call_args = vec![null_env];
                    for arg in args {
                        let v = compile_expr(ctx, arg)
                            .unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
                        let (tag, data) = to_tokvalue(ctx, v, &arg.ty);
                        call_args.push(tag);
                        call_args.push(data);
                    }
                    let call = ctx.builder.ins().call(func_ref, &call_args);
                    let results = ctx.builder.inst_results(call);
                    return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
                }
            }
        }
    }

    // Check if this is a call to a known function name
    if let HirExprKind::Ident(name) = &func_expr.kind {
        // User-defined functions take priority over builtins
        if ctx.compiler.declared_funcs.contains_key(name.as_str()) {
            // Try inlining small user functions (single-expression body, non-recursive)
            if can_inline_user_func(ctx, name) {
                if let Some(result) = compile_inline_user_func(ctx, name, args, result_ty) {
                    return Some(result);
                }
            }
            return compile_user_func_call(ctx, name, args, result_ty);
        }
        // Built-in function calls
        if let Some(result) = compile_builtin_call(ctx, name, args, result_ty) {
            return Some(result);
        }
    }

    // Generic function call (through closure expression)
    let func_val = compile_expr(ctx, func_expr);
    if let Some(raw_val) = func_val {
        // If func expr is Any-typed, extract closure ptr from TokValue data field
        let closure_ptr = if matches!(
            &func_expr.ty,
            Type::Any | Type::Optional(_) | Type::Result(_)
        ) {
            ctx.builder
                .ins()
                .load(types::I64, MemFlags::trusted(), raw_val, 8)
        } else {
            raw_val
        };
        return compile_closure_call(ctx, closure_ptr, args, result_ty);
    }

    Some(ctx.builder.ins().iconst(types::I64, 0))
}

/// Compile a call to a built-in function. Returns Some(value) if handled, None to fall through.
fn compile_builtin_call(
    ctx: &mut FuncCtx,
    name: &str,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    match name {
        // ── I/O builtins ──────────────────────────────────────────
        "p" | "print" => compile_print_call(ctx, args, false),
        "pl" | "println" => compile_print_call(ctx, args, true),
        // ── Collection query builtins ─────────────────────────────
        "len" => compile_builtin_len(ctx, args),
        "min" if args.len() == 1 => {
            compile_builtin_1_tokvalue(ctx, &args[0], "tok_array_min", result_ty)
        }
        "max" if args.len() == 1 => {
            compile_builtin_1_tokvalue(ctx, &args[0], "tok_array_max", result_ty)
        }
        "sum" if args.len() == 1 => {
            compile_builtin_1_tokvalue(ctx, &args[0], "tok_array_sum", result_ty)
        }
        "pop" if args.len() == 1 => {
            compile_builtin_1_tokvalue(ctx, &args[0], "tok_array_pop", result_ty)
        }
        // ── Array mutation builtins ───────────────────────────────
        "push" if args.len() >= 2 => compile_builtin_push(ctx, args),
        "sort" if args.len() == 1 => compile_builtin_1_ptr(ctx, &args[0], "tok_array_sort"),
        "rev" if args.len() == 1 => compile_builtin_1_ptr(ctx, &args[0], "tok_array_rev"),
        "flat" if args.len() == 1 => compile_builtin_1_ptr(ctx, &args[0], "tok_array_flat"),
        "uniq" if args.len() == 1 => compile_builtin_1_ptr(ctx, &args[0], "tok_array_uniq"),
        "freq" if args.len() == 1 => compile_builtin_1_ptr(ctx, &args[0], "tok_array_freq"),
        "zip" if args.len() >= 2 => compile_builtin_2_ptr(ctx, args, "tok_array_zip"),
        "slice" if args.len() >= 3 => compile_builtin_slice(ctx, args, result_ty),
        "pmap" if args.len() >= 2 => compile_builtin_pmap(ctx, args),
        // ── String builtins ──────────────────────────────────────
        "join" if args.len() >= 2 => compile_builtin_2_ptr(ctx, args, "tok_array_join"),
        "split" if args.len() >= 2 => compile_builtin_2_ptr(ctx, args, "tok_string_split"),
        "trim" if args.len() == 1 => compile_builtin_1_ptr(ctx, &args[0], "tok_string_trim"),
        // ── Map builtins ─────────────────────────────────────────
        "keys" if args.len() == 1 => compile_builtin_1_ptr(ctx, &args[0], "tok_map_keys"),
        "vals" if args.len() == 1 => compile_builtin_1_ptr(ctx, &args[0], "tok_map_vals"),
        "has" if args.len() >= 2 => compile_builtin_2_ptr(ctx, args, "tok_map_has"),
        "del" if args.len() >= 2 => compile_builtin_2_ptr(ctx, args, "tok_map_del"),
        "top" if args.len() >= 2 => compile_builtin_2_ptr(ctx, args, "tok_map_top"),
        // ── Type conversion builtins ─────────────────────────────
        "int" => compile_builtin_int(ctx, args),
        "float" => compile_builtin_float(ctx, args),
        "str" => compile_builtin_str(ctx, args),
        // ── Math builtins ────────────────────────────────────────
        "abs" => compile_builtin_abs(ctx, args, result_ty),
        "floor" => compile_builtin_floor_ceil(ctx, args, result_ty, "tok_value_floor", "tok_floor"),
        "ceil" => compile_builtin_floor_ceil(ctx, args, result_ty, "tok_value_ceil", "tok_ceil"),
        "rand" => {
            let func_ref = ctx.get_runtime_func_ref("tok_rand");
            let call = ctx.builder.ins().call(func_ref, &[]);
            Some(ctx.builder.inst_results(call)[0])
        }
        "clock" => {
            let func_ref = ctx.get_runtime_func_ref("tok_clock");
            let call = ctx.builder.ins().call(func_ref, &[]);
            Some(ctx.builder.inst_results(call)[0])
        }
        // ── System / concurrency builtins ────────────────────────
        "exit" => {
            if let Some(arg) = args.first() {
                let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
                // Unwrap Any-typed arg to get the raw i64 exit code
                let code = if matches!(arg.ty, Type::Any) {
                    from_tokvalue_raw_data(ctx, val)
                } else {
                    val
                };
                let func_ref = ctx.get_runtime_func_ref("tok_exit");
                ctx.builder.ins().call(func_ref, &[code]);
            }
            // exit never returns, but emit a dummy to satisfy the match
            Some(ctx.builder.ins().iconst(types::I64, 0))
        }
        "chan" => {
            let cap = if let Some(arg) = args.first() {
                let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
                // Unwrap Any-typed arg to get the raw i64 capacity
                if matches!(arg.ty, Type::Any) {
                    from_tokvalue_raw_data(ctx, val)
                } else {
                    val
                }
            } else {
                ctx.builder.ins().iconst(types::I64, 0)
            };
            let func_ref = ctx.get_runtime_func_ref("tok_channel_alloc");
            let call = ctx.builder.ins().call(func_ref, &[cap]);
            Some(ctx.builder.inst_results(call)[0])
        }
        "args" => {
            let func_ref = ctx.get_runtime_func_ref("tok_args");
            let call = ctx.builder.ins().call(func_ref, &[]);
            Some(ctx.builder.inst_results(call)[0])
        }
        "env" if args.len() == 1 => compile_builtin_1_tokvalue(ctx, &args[0], "tok_env", result_ty),
        // ── Introspection builtins ───────────────────────────────
        "type" => {
            if let Some(arg) = args.first() {
                let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
                let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
                let func_ref = ctx.get_runtime_func_ref("tok_type_of");
                let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                Some(ctx.builder.inst_results(call)[0])
            } else {
                None
            }
        }
        "is" if args.len() >= 2 => compile_builtin_is(ctx, args),
        // ── Known closure / variable call ────────────────────────
        _ => compile_known_closure_or_var_call(ctx, name, args, result_ty),
    }
}

/// Compile builtin `len(x)`.
fn compile_builtin_len(ctx: &mut FuncCtx, args: &[HirExpr]) -> Option<Value> {
    let arg = args.first()?;
    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
    match &arg.ty {
        Type::Any | Type::Optional(_) | Type::Result(_) => {
            let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
            let func_ref = ctx.get_runtime_func_ref("tok_value_len");
            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
            Some(ctx.builder.inst_results(call)[0])
        }
        _ => {
            let func_name = match &arg.ty {
                Type::Array(_) => "tok_array_len",
                Type::Str => "tok_string_len",
                Type::Map(_) => "tok_map_len",
                Type::Tuple(_) => "tok_tuple_len",
                _ => return Some(ctx.builder.ins().iconst(types::I64, 0)),
            };
            let func_ref = ctx.get_runtime_func_ref(func_name);
            let call = ctx.builder.ins().call(func_ref, &[val]);
            Some(ctx.builder.inst_results(call)[0])
        }
    }
}

/// Compile variadic `push(arr, v1, v2, ...)`.
fn compile_builtin_push(ctx: &mut FuncCtx, args: &[HirExpr]) -> Option<Value> {
    let mut arr = compile_expr_as_ptr(ctx, &args[0]);
    let func_ref = ctx.get_runtime_func_ref("tok_array_push");
    for arg in &args[1..] {
        let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
        let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
        let call = ctx.builder.ins().call(func_ref, &[arr, tag, data]);
        arr = ctx.builder.inst_results(call)[0];
    }
    Some(arr)
}

/// Compile `int(x)` type conversion.
fn compile_builtin_int(ctx: &mut FuncCtx, args: &[HirExpr]) -> Option<Value> {
    let arg = args.first()?;
    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
    if matches!(arg.ty, Type::Int) {
        return Some(val);
    }
    if matches!(arg.ty, Type::Float) {
        return Some(ctx.builder.ins().fcvt_to_sint_sat(types::I64, val));
    }
    let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
    let func_ref = ctx.get_runtime_func_ref("tok_to_int");
    let call = ctx.builder.ins().call(func_ref, &[tag, data]);
    Some(ctx.builder.inst_results(call)[0])
}

/// Compile `float(x)` type conversion.
fn compile_builtin_float(ctx: &mut FuncCtx, args: &[HirExpr]) -> Option<Value> {
    let arg = args.first()?;
    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
    if matches!(arg.ty, Type::Float) {
        return Some(val);
    }
    if matches!(arg.ty, Type::Int) {
        return Some(ctx.builder.ins().fcvt_from_sint(types::F64, val));
    }
    let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
    let func_ref = ctx.get_runtime_func_ref("tok_to_float");
    let call = ctx.builder.ins().call(func_ref, &[tag, data]);
    Some(ctx.builder.inst_results(call)[0])
}

/// Compile `str(x)` type conversion.
fn compile_builtin_str(ctx: &mut FuncCtx, args: &[HirExpr]) -> Option<Value> {
    let arg = args.first()?;
    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
    let func_name = match &arg.ty {
        Type::Int => "tok_int_to_string",
        Type::Float => "tok_float_to_string",
        Type::Bool => "tok_bool_to_string",
        Type::Str => return Some(val),
        _ => {
            let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
            let func_ref = ctx.get_runtime_func_ref("tok_value_to_string");
            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
    };
    let func_ref = ctx.get_runtime_func_ref(func_name);
    let call = ctx.builder.ins().call(func_ref, &[val]);
    Some(ctx.builder.inst_results(call)[0])
}

/// Compile `abs(x)` with type-specific dispatch.
fn compile_builtin_abs(ctx: &mut FuncCtx, args: &[HirExpr], result_ty: &Type) -> Option<Value> {
    let arg = args.first()?;
    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
    if matches!(arg.ty, Type::Any) {
        let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
        let func_ref = ctx.get_runtime_func_ref("tok_value_abs");
        let call = ctx.builder.ins().call(func_ref, &[tag, data]);
        let results = ctx.builder.inst_results(call);
        return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
    }
    let is_float = matches!(arg.ty, Type::Float);
    let func_name = if is_float {
        "tok_abs_float"
    } else {
        "tok_abs_int"
    };
    let func_ref = ctx.get_runtime_func_ref(func_name);
    let call = ctx.builder.ins().call(func_ref, &[val]);
    let raw = ctx.builder.inst_results(call)[0];
    if matches!(result_ty, Type::Any) {
        let tag_val = ctx
            .builder
            .ins()
            .iconst(types::I64, if is_float { TAG_FLOAT } else { TAG_INT });
        let data_val = if is_float {
            ctx.builder.ins().bitcast(types::I64, MemFlags::new(), raw)
        } else {
            raw
        };
        return Some(alloc_tokvalue_on_stack(ctx, tag_val, data_val));
    }
    Some(raw)
}

/// Compile `floor(x)` or `ceil(x)` with type-specific dispatch.
fn compile_builtin_floor_ceil(
    ctx: &mut FuncCtx,
    args: &[HirExpr],
    result_ty: &Type,
    any_func: &str,
    typed_func: &str,
) -> Option<Value> {
    let arg = args.first()?;
    let val = compile_expr(ctx, arg).expect("codegen: arg produced no value");
    if matches!(arg.ty, Type::Any) {
        let (tag, data) = to_tokvalue(ctx, val, &arg.ty);
        let func_ref = ctx.get_runtime_func_ref(any_func);
        let call = ctx.builder.ins().call(func_ref, &[tag, data]);
        let results = ctx.builder.inst_results(call);
        return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
    }
    let func_ref = ctx.get_runtime_func_ref(typed_func);
    let call = ctx.builder.ins().call(func_ref, &[val]);
    let raw = ctx.builder.inst_results(call)[0];
    if matches!(result_ty, Type::Any) {
        let tag_val = ctx.builder.ins().iconst(types::I64, TAG_INT);
        return Some(alloc_tokvalue_on_stack(ctx, tag_val, raw));
    }
    Some(raw)
}

/// Compile `slice(target, start, end)`.
fn compile_builtin_slice(ctx: &mut FuncCtx, args: &[HirExpr], result_ty: &Type) -> Option<Value> {
    let target_raw = compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
    let start = compile_expr(ctx, &args[1]).expect("codegen: args[1] produced no value");
    let end = compile_expr(ctx, &args[2]).expect("codegen: args[2] produced no value");
    if matches!(args[0].ty, Type::Any) {
        let (tag, data) = to_tokvalue(ctx, target_raw, &args[0].ty);
        let func_ref = ctx.get_runtime_func_ref("tok_value_slice");
        let call = ctx.builder.ins().call(func_ref, &[tag, data, start, end]);
        let results = ctx.builder.inst_results(call);
        return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
    }
    let target = unwrap_any_ptr(ctx, target_raw, &args[0].ty);
    let (func_name, tag_const) = match &args[0].ty {
        Type::Array(_) => ("tok_array_slice", TAG_ARRAY),
        Type::Str => ("tok_string_slice", TAG_STRING),
        _ => return None,
    };
    let func_ref = ctx.get_runtime_func_ref(func_name);
    let call = ctx.builder.ins().call(func_ref, &[target, start, end]);
    let raw = ctx.builder.inst_results(call)[0];
    if matches!(result_ty, Type::Any) {
        let tag = ctx.builder.ins().iconst(types::I64, tag_const);
        return Some(alloc_tokvalue_on_stack(ctx, tag, raw));
    }
    Some(raw)
}

/// Compile `pmap(arr, closure)`.
fn compile_builtin_pmap(ctx: &mut FuncCtx, args: &[HirExpr]) -> Option<Value> {
    let arr = compile_expr_as_ptr(ctx, &args[0]);
    let closure_ptr = compile_expr_as_ptr(ctx, &args[1]);
    let func_ref = ctx.get_runtime_func_ref("tok_pmap");
    let call = ctx.builder.ins().call(func_ref, &[arr, closure_ptr]);
    Some(ctx.builder.inst_results(call)[0])
}

/// Compile `is(val, type_str)`.
fn compile_builtin_is(ctx: &mut FuncCtx, args: &[HirExpr]) -> Option<Value> {
    let val_opt = compile_expr(ctx, &args[0]);
    let (tag, data) = if let Some(val) = val_opt {
        to_tokvalue(ctx, val, &args[0].ty)
    } else {
        let tag = ctx.builder.ins().iconst(types::I64, 0);
        let data = ctx.builder.ins().iconst(types::I64, 0);
        (tag, data)
    };
    let str_ptr = compile_expr_as_ptr(ctx, &args[1]);
    let func_ref = ctx.get_runtime_func_ref("tok_is");
    let call = ctx.builder.ins().call(func_ref, &[tag, data, str_ptr]);
    Some(ctx.builder.inst_results(call)[0])
}

/// Try calling a known closure or variable-held closure. Returns None to fall through.
fn compile_known_closure_or_var_call(
    ctx: &mut FuncCtx,
    name: &str,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    if let Some(kc) = ctx.known_closures.get(name).cloned() {
        let arg_types: Vec<Type> = args.iter().map(|a| a.ty.clone()).collect();
        let all_concrete = arg_types
            .iter()
            .all(|t| matches!(t, Type::Int | Type::Float | Type::Bool));
        if all_concrete {
            if can_inline_closure_call(
                &ctx.compiler.pending_lambdas[kc.pending_idx],
                &arg_types,
                name,
            ) {
                return compile_inline_closure_call(ctx, name, &kc, args, &arg_types, result_ty);
            }
            return compile_specialized_closure_call(ctx, name, &kc, args, &arg_types, result_ty);
        }
        return compile_direct_closure_call(ctx, kc.func_id, kc.env_ptr, args, result_ty);
    }
    if let Some((var, var_ty)) = ctx.vars.get(name).cloned() {
        if matches!(var_ty, Type::Func(_)) {
            let closure_ptr = ctx.builder.use_var(var);
            return compile_closure_call(ctx, closure_ptr, args, result_ty);
        }
        if matches!(var_ty, Type::Any) {
            let tokval_ptr = ctx.builder.use_var(var);
            let closure_ptr =
                ctx.builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), tokval_ptr, 8);
            return compile_closure_call(ctx, closure_ptr, args, result_ty);
        }
    }
    None
}

/// Check if a user-defined function can be inlined at the call site.
/// Eligible: single-expression or single-return body, non-recursive, small.
pub(crate) fn can_inline_user_func(ctx: &FuncCtx, name: &str) -> bool {
    let (params, _ret_type, body) = match ctx.compiler.func_bodies.get(name) {
        Some(v) => v,
        None => return false,
    };
    // Must be a single statement
    if body.len() != 1 {
        return false;
    }
    // All parameters must be scalar types or known closures
    // (skip inlining for Tuple/Map/Array params which have complex ABI)
    if !params
        .iter()
        .all(|p| matches!(p.ty, Type::Int | Type::Float | Type::Bool | Type::Any))
    {
        return false;
    }
    let expr = match &body[0] {
        HirStmt::Expr(e) => e,
        HirStmt::Return(Some(e)) => e,
        _ => return false,
    };
    // Don't inline self-recursive functions
    if contains_self_call(expr, name) {
        return false;
    }
    // Don't inline functions that contain embedded Return statements
    // (e.g., from ?^ error propagation or cond?^expr)
    // because Returns would jump to the wrong return block when inlined.
    if expr_contains_return(expr) {
        return false;
    }
    // Don't inline functions whose body is a side-effect-only expression (no return value).
    // The inlining path returns None for these, causing the call to also go through the
    // non-inlined path, executing side effects twice.
    if matches!(expr.kind, HirExprKind::Send { .. }) {
        return false;
    }
    true
}

/// Check if an HIR expression tree contains any Return statements
/// (nested in If/Block/Loop etc.)
pub(crate) fn expr_contains_return(expr: &HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => {
            expr_contains_return(cond)
                || then_body.iter().any(stmt_contains_return)
                || then_expr.as_ref().is_some_and(|e| expr_contains_return(e))
                || else_body.iter().any(stmt_contains_return)
                || else_expr.as_ref().is_some_and(|e| expr_contains_return(e))
        }
        HirExprKind::Block { stmts, expr: e } => {
            stmts.iter().any(stmt_contains_return)
                || e.as_ref().is_some_and(|e| expr_contains_return(e))
        }
        HirExprKind::BinOp { left, right, .. } => {
            expr_contains_return(left) || expr_contains_return(right)
        }
        HirExprKind::UnaryOp { operand, .. } => expr_contains_return(operand),
        HirExprKind::Call { func, args } => {
            expr_contains_return(func) || args.iter().any(expr_contains_return)
        }
        HirExprKind::Index { target, index } => {
            expr_contains_return(target) || expr_contains_return(index)
        }
        HirExprKind::Member { target, .. } => expr_contains_return(target),
        HirExprKind::Loop { body, .. } => body.iter().any(stmt_contains_return),
        HirExprKind::Array(elems) => elems.iter().any(expr_contains_return),
        HirExprKind::Tuple(elems) => elems.iter().any(expr_contains_return),
        _ => false,
    }
}

pub(crate) fn stmt_contains_return(stmt: &HirStmt) -> bool {
    match stmt {
        HirStmt::Return(_) => true,
        HirStmt::Expr(e) => expr_contains_return(e),
        HirStmt::Assign { value, .. } => expr_contains_return(value),
        _ => false,
    }
}

/// Inline a user-defined function at the call site.
pub(crate) fn compile_inline_user_func(
    ctx: &mut FuncCtx,
    name: &str,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    let (params, _ret_type, body) = ctx.compiler.func_bodies.get(name).cloned()?;
    if params.len() != args.len() {
        return None;
    }

    // Compile argument expressions first
    let mut arg_vals = Vec::new();
    let mut arg_types = Vec::new();
    for arg in args {
        let v = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        arg_vals.push(v);
        arg_types.push(arg.ty.clone());
    }

    // Save old bindings and bind function parameters to arg values.
    // Also propagate known closures from call site to inlined body.
    let mut old_bindings: Vec<(String, Option<(Variable, Type)>)> = Vec::new();
    let mut old_kc_bindings: Vec<(String, Option<KnownClosure>)> = Vec::new();
    for (i, param) in params.iter().enumerate() {
        old_bindings.push((param.name.clone(), ctx.vars.remove(&param.name)));
        // If param is Any and arg is concrete, keep as concrete type
        let actual_ty = if matches!(param.ty, Type::Any) {
            &arg_types[i]
        } else {
            &param.ty
        };
        let ct = cl_type_or_i64(actual_ty);
        let var = ctx.new_var(ct);
        // Coerce arg value to the parameter type if needed
        let coerced = if matches!(param.ty, Type::Any) && !matches!(arg_types[i], Type::Any) {
            // Parameter is Any but we keep concrete type (no boxing needed when inlining)
            arg_vals[i]
        } else {
            coerce_value(ctx, arg_vals[i], &arg_types[i], actual_ty)
        };
        ctx.builder.def_var(var, coerced);
        ctx.vars
            .insert(param.name.clone(), (var, actual_ty.clone()));

        // Propagate known closure info: if the argument is a known closure variable,
        // bind that closure info to the parameter name so fn(x) can be inlined/specialized
        if let HirExprKind::Ident(arg_name) = &args[i].kind {
            if let Some(kc) = ctx.known_closures.get(arg_name).cloned() {
                old_kc_bindings.push((param.name.clone(), ctx.known_closures.remove(&param.name)));
                ctx.known_closures.insert(param.name.clone(), kc);
            }
        }
    }

    // Retype the body with concrete arg types
    let mut type_map = HashMap::new();
    for (i, param) in params.iter().enumerate() {
        let actual_ty = if matches!(param.ty, Type::Any) {
            &arg_types[i]
        } else {
            &param.ty
        };
        type_map.insert(param.name.clone(), actual_ty.clone());
    }
    let retyped = retype_body(&body, &type_map);
    let retyped = unwrap_return_stmts(retyped);

    // Determine result type from retyped body
    let body_result_ty = retyped
        .last()
        .and_then(|s| match s {
            HirStmt::Expr(e) => Some(e.ty.clone()),
            _ => None,
        })
        .unwrap_or(Type::Nil);

    // Compile the body inline
    let body_result = compile_body(ctx, &retyped, &body_result_ty);

    // Restore old bindings
    for (pname, old) in old_bindings {
        ctx.vars.remove(&pname);
        if let Some(old_val) = old {
            ctx.vars.insert(pname, old_val);
        }
    }
    // Restore old known_closure bindings
    for (pname, old) in old_kc_bindings {
        ctx.known_closures.remove(&pname);
        if let Some(old_val) = old {
            ctx.known_closures.insert(pname, old_val);
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

pub(crate) fn compile_user_func_call(
    ctx: &mut FuncCtx,
    name: &str,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    // TCO: if this is a tail call to self, compile args and jump to loop header
    if ctx.tco_func_name.as_deref() == Some(name) {
        if let Some(loop_header) = ctx.tco_loop_header {
            let func_sig = ctx.compiler.func_sigs.get(name).cloned();
            let mut jump_vals = Vec::new();
            for (i, arg) in args.iter().enumerate() {
                let param_ty = func_sig
                    .as_ref()
                    .and_then(|s| s.0.get(i))
                    .cloned()
                    .unwrap_or(arg.ty.clone());
                if let Some(v) = compile_expr(ctx, arg) {
                    if matches!(param_ty, Type::Any) {
                        let (tag, data) = to_tokvalue(ctx, v, &arg.ty);
                        jump_vals.push(tag);
                        jump_vals.push(data);
                    } else if matches!(arg.ty, Type::Any) {
                        let coerced = coerce_value(ctx, v, &arg.ty, &param_ty);
                        jump_vals.push(coerced);
                    } else {
                        jump_vals.push(v);
                    }
                } else {
                    // Nil-typed arg: push zero value(s) to match block params
                    if matches!(param_ty, Type::Any) {
                        let zero = ctx.builder.ins().iconst(types::I64, 0);
                        jump_vals.push(zero);
                        jump_vals.push(zero);
                    } else {
                        let ct = cl_type_or_i64(&param_ty);
                        let zero = zero_value(&mut ctx.builder, ct);
                        jump_vals.push(zero);
                    }
                }
            }
            ctx.builder.ins().jump(loop_header, &jump_vals);
            // Create a dead block for any unreachable code after the tail call
            let dead_block = ctx.builder.create_block();
            ctx.builder.switch_to_block(dead_block);
            ctx.builder.seal_block(dead_block);
            ctx.block_terminated = true;
            return None;
        }
    }

    let func_sig = ctx.compiler.func_sigs.get(name).cloned();
    let mut arg_vals = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        if let Some(v) = compile_expr(ctx, arg) {
            let param_ty = func_sig
                .as_ref()
                .and_then(|s| s.0.get(i))
                .cloned()
                .unwrap_or(arg.ty.clone());
            if matches!(param_ty, Type::Any) {
                // Any param: always pass as (tag, data) pair
                let (tag, data) = to_tokvalue(ctx, v, &arg.ty);
                arg_vals.push(tag);
                arg_vals.push(data);
            } else if matches!(arg.ty, Type::Any) {
                // Any → Concrete param: extract from TokValue ptr
                let coerced = coerce_value(ctx, v, &arg.ty, &param_ty);
                arg_vals.push(coerced);
            } else {
                arg_vals.push(v);
            }
        }
    }
    let func_ref = ctx.get_tok_func_ref(name);
    let call = ctx.builder.ins().call(func_ref, &arg_vals);
    let results = ctx.builder.inst_results(call);
    if results.is_empty() {
        return None;
    }
    let ret_ty = func_sig
        .as_ref()
        .map(|s| s.1.clone())
        .unwrap_or(result_ty.clone());
    if matches!(ret_ty, Type::Any) {
        // Any return: 2 results (tag, data) — pack into stack TokValue
        let tag = results[0];
        let data = results[1];
        return Some(alloc_tokvalue_on_stack(ctx, tag, data));
    }
    Some(coerce_value(ctx, results[0], &ret_ty, result_ty))
}

/// Call a closure indirectly: extract fn_ptr and env_ptr, build signature, call_indirect.
pub(crate) fn compile_closure_call(
    ctx: &mut FuncCtx,
    closure_ptr: Value,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    // Extract fn_ptr and env_ptr directly from TokClosure struct (repr(C)):
    //   +0: rc (AtomicU32, 4B) + padding (4B)
    //   +8: fn_ptr (*const u8, 8B)
    //   +16: env_ptr (*mut u8, 8B)
    let fn_ptr = ctx
        .builder
        .ins()
        .load(PTR, MemFlags::trusted(), closure_ptr, 8);
    let env_ptr = ctx
        .builder
        .ins()
        .load(PTR, MemFlags::trusted(), closure_ptr, 16);

    // Build or reuse cached signature for indirect call: (env: PTR, tag0: I64, data0: I64, ...) -> (I64, I64)
    let n_args = args.len();
    let sig_ref = if let Some(&cached) = ctx.closure_sig_cache.get(&n_args) {
        cached
    } else {
        let mut sig = ctx.compiler.module.make_signature();
        sig.params.push(AbiParam::new(PTR)); // env
        for _ in 0..n_args {
            sig.params.push(AbiParam::new(types::I64)); // tag
            sig.params.push(AbiParam::new(types::I64)); // data
        }
        sig.returns.push(AbiParam::new(types::I64)); // ret tag
        sig.returns.push(AbiParam::new(types::I64)); // ret data
        let sr = ctx.builder.import_signature(sig);
        ctx.closure_sig_cache.insert(n_args, sr);
        sr
    };

    // Build args: env, then (tag, data) pairs for each arg
    let mut call_args = vec![env_ptr];
    for arg in args {
        let v = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        let (tag, data) = to_tokvalue(ctx, v, &arg.ty);
        call_args.push(tag);
        call_args.push(data);
    }

    let call = ctx.builder.ins().call_indirect(sig_ref, fn_ptr, &call_args);
    let results = ctx.builder.inst_results(call);
    Some(from_tokvalue(ctx, results[0], results[1], result_ty))
}

/// Call a closure directly when we know the FuncId at compile time.
/// Still uses the uniform (tag, data) calling convention, but avoids call_indirect.
fn compile_direct_closure_call(
    ctx: &mut FuncCtx,
    func_id: FuncId,
    env_ptr: Value,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    let func_ref = ctx
        .compiler
        .module
        .declare_func_in_func(func_id, ctx.builder.func);

    // Build args: env, then (tag, data) pairs for each arg
    let mut call_args = vec![env_ptr];
    for arg in args {
        let v = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        let (tag, data) = to_tokvalue(ctx, v, &arg.ty);
        call_args.push(tag);
        call_args.push(data);
    }

    let call = ctx.builder.ins().call(func_ref, &call_args);
    let results = ctx.builder.inst_results(call);
    Some(from_tokvalue(ctx, results[0], results[1], result_ty))
}

/// Call a closure with a type-specialized calling convention (native types, no boxing).
/// Lazily creates the specialized function on first call.
fn compile_specialized_closure_call(
    ctx: &mut FuncCtx,
    name: &str,
    kc: &KnownClosure,
    args: &[HirExpr],
    arg_types: &[Type],
    result_ty: &Type,
) -> Option<Value> {
    // Check if we already have a specialized version for these arg types
    let existing = if let Some((sid, ref stypes, ref sret)) = kc.specialized {
        if stypes == arg_types {
            Some((sid, sret.clone()))
        } else {
            None
        }
    } else {
        None
    };

    let (spec_func_id, spec_ret_type) = if let Some(pair) = existing {
        pair
    } else {
        // Create specialized function
        let orig = &ctx.compiler.pending_lambdas[kc.pending_idx];
        let spec_name = format!("{}_spec", orig.name);

        // Build type map and retype body FIRST so we can get the accurate return type
        let mut type_map = HashMap::new();
        for (param, at) in orig.params.iter().zip(arg_types.iter()) {
            type_map.insert(param.name.clone(), at.clone());
        }
        for cap in &orig.captures {
            if matches!(cap.ty, Type::Int | Type::Float | Type::Bool) {
                type_map.insert(cap.name.clone(), cap.ty.clone());
            }
        }
        let retyped_body = retype_body(&orig.body, &type_map);

        // Derive return type from the retyped body's last expression
        let ret_type = retyped_body
            .last()
            .and_then(|s| match s {
                HirStmt::Expr(e) => {
                    if matches!(e.ty, Type::Any) {
                        None
                    } else {
                        Some(e.ty.clone())
                    }
                }
                _ => None,
            })
            .unwrap_or(Type::Int); // fallback for simple arithmetic lambdas

        // Build signature: (env: PTR, arg0: T0, ...) -> RetT
        let mut sig = ctx.compiler.module.make_signature();
        sig.params.push(AbiParam::new(PTR)); // env_ptr
        for at in arg_types {
            sig.params.push(AbiParam::new(cl_type_or_i64(at)));
        }
        sig.returns.push(AbiParam::new(cl_type_or_i64(&ret_type)));

        let func_id = ctx
            .compiler
            .module
            .declare_function(&spec_name, Linkage::Local, &sig)
            .expect("codegen: failed to declare specialized lambda");

        // Create specialized PendingLambda with retyped body
        ctx.compiler.pending_lambdas.push(PendingLambda {
            name: spec_name,
            func_id,
            params: orig.params.clone(),
            ret_type: ret_type.clone(),
            body: retyped_body,
            captures: orig.captures.clone(),
            specialized_param_types: Some(arg_types.to_vec()),
        });

        // Update known_closures with specialized info
        if let Some(kc_mut) = ctx.known_closures.get_mut(name) {
            kc_mut.specialized = Some((func_id, arg_types.to_vec(), ret_type.clone()));
        }

        (func_id, ret_type)
    };

    let env_ptr = kc.env_ptr;
    let func_ref = ctx
        .compiler
        .module
        .declare_func_in_func(spec_func_id, ctx.builder.func);

    // Build args with native types (no boxing)
    let mut call_args = vec![env_ptr];
    for arg in args {
        let v = compile_expr(ctx, arg).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
        call_args.push(v);
    }

    let call = ctx.builder.ins().call(func_ref, &call_args);
    let results = ctx.builder.inst_results(call);
    if results.is_empty() {
        return None;
    }
    let raw_val = results[0];

    // If caller expects Any but we returned a concrete native type, wrap as TokValue
    if matches!(result_ty, Type::Any) && !matches!(spec_ret_type, Type::Any) {
        let (tag, data) = to_tokvalue(ctx, raw_val, &spec_ret_type);
        Some(alloc_tokvalue_on_stack(ctx, tag, data))
    } else {
        Some(coerce_value(ctx, raw_val, &spec_ret_type, result_ty))
    }
}
