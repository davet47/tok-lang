// ─── Expression compilation ───────────────────────────────────────────

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_module::{Linkage, Module};

use tok_hir::hir::*;
use tok_types::Type;

use super::{
    alloc_tokvalue_on_stack, cl_type, coerce_value, compile_expr_as_ptr, compile_stmt,
    from_tokvalue, to_tokvalue, unwrap_any_ptr, FuncCtx, PendingLambda, PTR, TAG_ARRAY,
};
use super::{can_inline_hof, compile_inline_filter, compile_inline_reduce};
use super::{
    compile_binop, compile_call, compile_go_expr, compile_if, compile_lambda_expr, compile_loop,
    compile_receive_expr, compile_select_expr, compile_unaryop,
};
use super::{get_stdlib_const, stdlib_constructor};

/// Compile an HIR expression into Cranelift IR, returning the resulting value.
///
/// # Return value
///
/// - `Some(val)`: The expression produced a Cranelift `Value`. For concrete types
///   (Int, Float, Bool, Str, Array, Map, etc.) this is the native representation.
///   For `Type::Any` this is a pointer to a stack-allocated TokValue.
/// - `None`: The expression is `Type::Nil` and produces no runtime value (e.g., a
///   bare function call used as a statement, `print(x)`). Callers must handle `None`
///   gracefully — typically by substituting `iconst(0)` when a value is required.
///
/// # Panics
///
/// Panics if the expression kind is not recognized by the compiler.
pub(crate) fn compile_expr(ctx: &mut FuncCtx, expr: &HirExpr) -> Option<Value> {
    match &expr.kind {
        HirExprKind::Int(n) => Some(ctx.builder.ins().iconst(types::I64, *n)),

        HirExprKind::Float(f) => Some(ctx.builder.ins().f64const(*f)),

        HirExprKind::Bool(b) => Some(ctx.builder.ins().iconst(types::I8, *b as i64)),

        HirExprKind::Nil => {
            // Return a zero i64 as a nil sentinel in contexts that need a value
            if cl_type(&expr.ty).is_some() {
                Some(ctx.builder.ins().iconst(types::I64, 0))
            } else {
                None
            }
        }

        HirExprKind::Str(s) => {
            let (data_id, len) = ctx.compiler.declare_string_data(s);
            let gv = ctx.get_data_ref(data_id);
            let ptr = ctx.builder.ins().global_value(PTR, gv);
            let len_val = ctx.builder.ins().iconst(types::I64, len as i64);
            let func_ref = ctx.get_runtime_func_ref("tok_string_alloc");
            let call = ctx.builder.ins().call(func_ref, &[ptr, len_val]);
            Some(ctx.builder.inst_results(call)[0])
        }

        HirExprKind::Ident(name) => {
            if let Some((var, var_ty)) = ctx.vars.get(name).cloned() {
                let raw = ctx.builder.use_var(var);
                // If the variable is stored as Any but the expression type is concrete,
                // coerce from Any → concrete. This happens with captured variables
                // in lambdas, whose vars are stored as Any (TokValue ptr) but the HIR
                // still has the original concrete type from type checking.
                if matches!(var_ty, Type::Any)
                    && !matches!(&expr.ty, Type::Any | Type::Nil | Type::Never)
                {
                    Some(coerce_value(ctx, raw, &Type::Any, &expr.ty))
                } else if !matches!(var_ty, Type::Any) && matches!(&expr.ty, Type::Any) {
                    // Variable stored as concrete (e.g., Int) but HIR thinks it's Any
                    // (happens when variable was first assigned concrete, then reassigned
                    // from an Any-typed expression — codegen keeps the original type).
                    // Wrap the concrete value into a TokValue pointer.
                    Some(coerce_value(ctx, raw, &var_ty, &Type::Any))
                } else {
                    Some(raw)
                }
            } else if ctx.compiler.declared_funcs.contains_key(name.as_str()) {
                // Declared function used as a value — create a trampoline wrapper
                // with the closure calling convention, then wrap in a TokClosure.
                let (param_types, ret_type) = ctx
                    .compiler
                    .func_sigs
                    .get(name.as_str())
                    .expect("codegen: unknown function signature")
                    .clone();
                let trampoline_name = format!("__tok_tramp_{}", name);

                // Check if trampoline already exists (avoid duplicate definition)
                let tramp_func_id = if let Some(&existing) =
                    ctx.compiler.declared_funcs.get(trampoline_name.as_str())
                {
                    existing
                } else {
                    // Create trampoline as a PendingLambda that just calls the function
                    let tramp_params: Vec<HirParam> = param_types
                        .iter()
                        .enumerate()
                        .map(|(i, ty)| HirParam {
                            name: format!("__p{}", i),
                            ty: ty.clone(),
                            variadic: false,
                            has_default: false,
                        })
                        .collect();
                    let call_args: Vec<HirExpr> = tramp_params
                        .iter()
                        .map(|p| HirExpr::new(HirExprKind::Ident(p.name.clone()), p.ty.clone()))
                        .collect();
                    let call_expr = HirExpr::new(
                        HirExprKind::Call {
                            func: Box::new(HirExpr::new(
                                HirExprKind::Ident(name.clone()),
                                Type::Any,
                            )),
                            args: call_args,
                        },
                        ret_type.clone(),
                    );

                    // Declare trampoline function signature: (env, tag0, data0, ...) -> (tag, data)
                    let mut sig = ctx.compiler.module.make_signature();
                    sig.params.push(AbiParam::new(PTR)); // env_ptr
                    for _ in &param_types {
                        sig.params.push(AbiParam::new(types::I64)); // tag
                        sig.params.push(AbiParam::new(types::I64)); // data
                    }
                    sig.returns.push(AbiParam::new(types::I64)); // result tag
                    sig.returns.push(AbiParam::new(types::I64)); // result data

                    let fid = ctx
                        .compiler
                        .module
                        .declare_function(&trampoline_name, Linkage::Local, &sig)
                        .expect("codegen: failed to declare trampoline");

                    ctx.compiler
                        .declared_funcs
                        .insert(trampoline_name.clone(), fid);

                    ctx.compiler.pending_lambdas.push(PendingLambda {
                        name: trampoline_name.clone(),
                        func_id: fid,
                        params: tramp_params,
                        ret_type: ret_type.clone(),
                        body: vec![HirStmt::Expr(call_expr)],
                        captures: vec![],
                        specialized_param_types: None,
                    });

                    fid
                };

                let tramp_ref = ctx
                    .compiler
                    .module
                    .declare_func_in_func(tramp_func_id, ctx.builder.func);
                let fn_ptr = ctx.builder.ins().func_addr(PTR, tramp_ref);
                let env_ptr = ctx.builder.ins().iconst(PTR, 0);
                let arity_val = ctx
                    .builder
                    .ins()
                    .iconst(types::I32, param_types.len() as i64);
                let env_count_val = ctx.builder.ins().iconst(types::I32, 0);
                let alloc_ref = ctx.get_runtime_func_ref("tok_closure_alloc");
                let call = ctx
                    .builder
                    .ins()
                    .call(alloc_ref, &[fn_ptr, env_ptr, arity_val, env_count_val]);
                Some(ctx.builder.inst_results(call)[0])
            } else {
                // Unknown variable — return 0 as fallback
                Some(ctx.builder.ins().iconst(types::I64, 0))
            }
        }

        HirExprKind::Array(elems) => {
            // Allocate empty array, then push each element
            let alloc_ref = ctx.get_runtime_func_ref("tok_array_alloc");
            let call = ctx.builder.ins().call(alloc_ref, &[]);
            let mut arr = ctx.builder.inst_results(call)[0];

            for elem in elems {
                let val = compile_expr(ctx, elem)
                    .unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
                let (tag, data) = to_tokvalue(ctx, val, &elem.ty);
                let push_ref = ctx.get_runtime_func_ref("tok_array_push");
                let push_call = ctx.builder.ins().call(push_ref, &[arr, tag, data]);
                arr = ctx.builder.inst_results(push_call)[0];
            }
            Some(arr)
        }

        HirExprKind::Map(entries) => {
            let alloc_ref = ctx.get_runtime_func_ref("tok_map_alloc");
            let call = ctx.builder.ins().call(alloc_ref, &[]);
            let map = ctx.builder.inst_results(call)[0];

            for (key, val_expr) in entries {
                // Allocate key string
                let (data_id, len) = ctx.compiler.declare_string_data(key);
                let gv = ctx.get_data_ref(data_id);
                let key_ptr = ctx.builder.ins().global_value(PTR, gv);
                let key_len = ctx.builder.ins().iconst(types::I64, len as i64);
                let str_ref = ctx.get_runtime_func_ref("tok_string_alloc");
                let str_call = ctx.builder.ins().call(str_ref, &[key_ptr, key_len]);
                let key_str = ctx.builder.inst_results(str_call)[0];

                let val = compile_expr(ctx, val_expr)
                    .unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
                let (tag, data) = to_tokvalue(ctx, val, &val_expr.ty);
                let set_ref = ctx.get_runtime_func_ref("tok_map_set");
                ctx.builder.ins().call(set_ref, &[map, key_str, tag, data]);
            }
            Some(map)
        }

        HirExprKind::Tuple(elems) => {
            let count = ctx.builder.ins().iconst(types::I64, elems.len() as i64);
            let alloc_ref = ctx.get_runtime_func_ref("tok_tuple_alloc");
            let call = ctx.builder.ins().call(alloc_ref, &[count]);
            let tuple = ctx.builder.inst_results(call)[0];

            for (i, elem) in elems.iter().enumerate() {
                let val = compile_expr(ctx, elem)
                    .unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
                let idx = ctx.builder.ins().iconst(types::I64, i as i64);
                let (tag, data) = to_tokvalue(ctx, val, &elem.ty);
                let set_ref = ctx.get_runtime_func_ref("tok_tuple_set");
                ctx.builder.ins().call(set_ref, &[tuple, idx, tag, data]);
            }
            Some(tuple)
        }

        HirExprKind::BinOp { op, left, right } => compile_binop(ctx, *op, left, right, &expr.ty),

        HirExprKind::UnaryOp { op, operand } => compile_unaryop(ctx, *op, operand, &expr.ty),

        HirExprKind::Index { target, index } => {
            let target_val =
                compile_expr(ctx, target).expect("codegen: target expr produced no value");
            let idx_val = compile_expr(ctx, index).expect("codegen: index expr produced no value");
            // Index must be a raw i64 integer for runtime calls — unwrap/widen as needed
            let idx_raw = {
                let actual = ctx.builder.func.dfg.value_type(idx_val);
                if actual == types::I8 {
                    // Bool used as index (e.g., from logical |) — widen to i64
                    ctx.builder.ins().uextend(types::I64, idx_val)
                } else if matches!(&index.ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                    // Any-typed TokValue pointer — extract integer
                    coerce_value(ctx, idx_val, &index.ty, &Type::Int)
                } else {
                    idx_val
                }
            };
            match &target.ty {
                Type::Array(_) => {
                    let func_ref = ctx.get_runtime_func_ref("tok_array_get");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, idx_raw]);
                    let results = ctx.builder.inst_results(call);
                    // Returns TokValue (tag, data) — extract based on expected type
                    let tag = results[0];
                    let data = results[1];
                    Some(from_tokvalue(ctx, tag, data, &expr.ty))
                }
                Type::Str => {
                    let func_ref = ctx.get_runtime_func_ref("tok_string_index");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, idx_raw]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                Type::Tuple(_) => {
                    let func_ref = ctx.get_runtime_func_ref("tok_tuple_get");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, idx_raw]);
                    let results = ctx.builder.inst_results(call);
                    let tag = results[0];
                    let data = results[1];
                    Some(from_tokvalue(ctx, tag, data, &expr.ty))
                }
                Type::Map(_) => {
                    // Index by string key — unwrap key from Any if needed
                    let key = unwrap_any_ptr(ctx, idx_val, &index.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_map_get");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, key]);
                    let results = ctx.builder.inst_results(call);
                    let tag = results[0];
                    let data = results[1];
                    Some(from_tokvalue(ctx, tag, data, &expr.ty))
                }
                Type::Any | Type::Optional(_) | Type::Result(_) => {
                    // Extract (tag, data) from TokValue pointer, call tok_value_index
                    let (t_tag, t_data) = to_tokvalue(ctx, target_val, &target.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_value_index");
                    let call = ctx.builder.ins().call(func_ref, &[t_tag, t_data, idx_raw]);
                    let results = ctx.builder.inst_results(call);
                    Some(from_tokvalue(ctx, results[0], results[1], &expr.ty))
                }
                _ => Some(target_val),
            }
        }

        HirExprKind::Member { target, field } => {
            // ── Stdlib constant inlining (e.g. m.pi, m.e, m.inf, m.nan) ──
            if let HirExprKind::Ident(var_name) = &target.kind {
                if let Some(module_name) = ctx.stdlib_imports.get(var_name.as_str()).cloned() {
                    if let Some(const_val) = get_stdlib_const(&module_name, field) {
                        let f_val = ctx.builder.ins().f64const(const_val);
                        let (tag, data) = to_tokvalue(ctx, f_val, &Type::Float);
                        return Some(from_tokvalue(ctx, tag, data, &expr.ty));
                    }
                }
            }
            let target_val =
                compile_expr(ctx, target).expect("codegen: target expr produced no value");

            // Tuple: numeric field access — no string allocation needed
            if matches!(&target.ty, Type::Tuple(_)) {
                if let Ok(idx) = field.parse::<i64>() {
                    let idx_val = ctx.builder.ins().iconst(types::I64, idx);
                    let func_ref = ctx.get_runtime_func_ref("tok_tuple_get");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, idx_val]);
                    let results = ctx.builder.inst_results(call);
                    return Some(from_tokvalue(ctx, results[0], results[1], &expr.ty));
                } else {
                    return Some(target_val);
                }
            }

            // Allocate field name as string for map access
            let (data_id, len) = ctx.compiler.declare_string_data(field);
            let gv = ctx.get_data_ref(data_id);
            let key_ptr = ctx.builder.ins().global_value(PTR, gv);
            let key_len = ctx.builder.ins().iconst(types::I64, len as i64);
            let str_ref = ctx.get_runtime_func_ref("tok_string_alloc");
            let str_call = ctx.builder.ins().call(str_ref, &[key_ptr, key_len]);
            let key_str = ctx.builder.inst_results(str_call)[0];

            let result = match &target.ty {
                Type::Any | Type::Optional(_) | Type::Result(_) => {
                    // target_val is a PTR to stack TokValue — extract the map pointer
                    let map_ptr =
                        ctx.builder
                            .ins()
                            .load(types::I64, MemFlags::trusted(), target_val, 8);
                    let func_ref = ctx.get_runtime_func_ref("tok_map_get");
                    let call = ctx.builder.ins().call(func_ref, &[map_ptr, key_str]);
                    let results = ctx.builder.inst_results(call);
                    from_tokvalue(ctx, results[0], results[1], &expr.ty)
                }
                _ => {
                    let func_ref = ctx.get_runtime_func_ref("tok_map_get");
                    let call = ctx.builder.ins().call(func_ref, &[target_val, key_str]);
                    let results = ctx.builder.inst_results(call);
                    from_tokvalue(ctx, results[0], results[1], &expr.ty)
                }
            };

            // Free temporary key string
            let free_ref = ctx.get_runtime_func_ref("tok_string_free");
            ctx.builder.ins().call(free_ref, &[key_str]);

            Some(result)
        }

        HirExprKind::Call { func, args } => compile_call(ctx, func, args, &expr.ty),

        HirExprKind::RuntimeCall { name, args } => compile_runtime_call(ctx, name, args, &expr.ty),

        HirExprKind::Lambda {
            params,
            ret_type,
            body,
        } => compile_lambda_expr(ctx, params, ret_type, body),

        HirExprKind::If {
            cond,
            then_body,
            then_expr,
            else_body,
            else_expr,
        } => compile_if(
            ctx, cond, then_body, then_expr, else_body, else_expr, &expr.ty,
        ),

        HirExprKind::Loop { kind, body } => {
            compile_loop(ctx, kind, body);
            None
        }

        HirExprKind::Block {
            stmts,
            expr: block_expr,
        } => {
            for s in stmts {
                compile_stmt(ctx, s);
            }
            if let Some(e) = block_expr {
                compile_expr(ctx, e)
            } else {
                None
            }
        }

        HirExprKind::Length(target) => {
            let target_val =
                compile_expr(ctx, target).expect("codegen: target expr produced no value");
            match &target.ty {
                Type::Array(_) => {
                    let func_ref = ctx.get_runtime_func_ref("tok_array_len");
                    let call = ctx.builder.ins().call(func_ref, &[target_val]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                Type::Str => {
                    let func_ref = ctx.get_runtime_func_ref("tok_string_len");
                    let call = ctx.builder.ins().call(func_ref, &[target_val]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                Type::Map(_) => {
                    let func_ref = ctx.get_runtime_func_ref("tok_map_len");
                    let call = ctx.builder.ins().call(func_ref, &[target_val]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                Type::Tuple(_) => {
                    let func_ref = ctx.get_runtime_func_ref("tok_tuple_len");
                    let call = ctx.builder.ins().call(func_ref, &[target_val]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                Type::Any | Type::Optional(_) | Type::Result(_) => {
                    // Any: target_val is a pointer to a stack-allocated TokValue
                    // Extract (tag, data) and call tok_value_len
                    let (tag, data) = to_tokvalue(ctx, target_val, &target.ty);
                    let func_ref = ctx.get_runtime_func_ref("tok_value_len");
                    let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                    Some(ctx.builder.inst_results(call)[0])
                }
                _ => Some(ctx.builder.ins().iconst(types::I64, 0)),
            }
        }

        HirExprKind::Range {
            start,
            end: _,
            inclusive: _,
        } => {
            // Ranges are only used in for-loops, which handle them directly.
            // If a range appears as a standalone expression, just return start.
            compile_expr(ctx, start)
        }

        HirExprKind::Go(body_expr) => compile_go_expr(ctx, body_expr),

        HirExprKind::Receive(chan_expr) => compile_receive_expr(ctx, chan_expr, &expr.ty),

        HirExprKind::Send { chan, value } => {
            let chan_val =
                compile_expr(ctx, chan).expect("codegen: channel expr produced no value");
            let val = compile_expr(ctx, value).expect("codegen: value expr produced no value");
            let (tag, data) = to_tokvalue(ctx, val, &value.ty);
            let func_ref = ctx.get_runtime_func_ref("tok_channel_send");
            ctx.builder.ins().call(func_ref, &[chan_val, tag, data]);
            None
        }

        HirExprKind::Select(arms) => compile_select_expr(ctx, arms),
    }
}

// ─── Extracted expression helpers ─────────────────────────────────────

/// Compile a RuntimeCall expression (imports, filter, reduce, push, concat, etc.).
pub(crate) fn compile_runtime_call(
    ctx: &mut FuncCtx,
    name: &str,
    args: &[HirExpr],
    result_ty: &Type,
) -> Option<Value> {
    match name {
        "tok_import" => {
            if let HirExprKind::Str(path) = &args[0].kind {
                let constructor = stdlib_constructor(path).unwrap_or_else(|| {
                    panic!(
                        "Unknown module: @\"{}\" — only stdlib modules are supported in compiled mode",
                        path
                    )
                });
                let func_ref = ctx.get_runtime_func_ref(constructor);
                let call = ctx.builder.ins().call(func_ref, &[]);
                return Some(ctx.builder.inst_results(call)[0]);
            }
            panic!("Dynamic imports not supported in compiled mode");
        }
        "tok_array_filter" => {
            if can_inline_hof(&args[1], &args[0].ty, 1) {
                return compile_inline_filter(ctx, &args[0], &args[1], result_ty);
            }
            let arr = compile_expr_as_ptr(ctx, &args[0]);
            let closure = compile_expr(ctx, &args[1]).expect("codegen: args[1] produced no value");
            let func_ref = ctx.get_runtime_func_ref("tok_array_filter");
            let call = ctx.builder.ins().call(func_ref, &[arr, closure]);
            let result = ctx.builder.inst_results(call)[0];
            if matches!(result_ty, Type::Any | Type::Optional(_) | Type::Result(_)) {
                let tag = ctx.builder.ins().iconst(types::I64, TAG_ARRAY);
                return Some(alloc_tokvalue_on_stack(ctx, tag, result));
            }
            return Some(result);
        }
        "tok_array_reduce" => {
            if can_inline_hof(&args[2], &args[0].ty, 2) {
                return compile_inline_reduce(ctx, &args[0], &args[1], &args[2], result_ty);
            }
            let arr = compile_expr_as_ptr(ctx, &args[0]);
            let init_val = compile_expr(ctx, &args[1]);
            let closure = compile_expr(ctx, &args[2]).expect("codegen: args[2] produced no value");
            let (init_tag, init_data) = if let Some(iv) = init_val {
                to_tokvalue(ctx, iv, &args[1].ty)
            } else {
                let zero = ctx.builder.ins().iconst(types::I64, 0);
                (zero, zero)
            };
            let func_ref = ctx.get_runtime_func_ref("tok_array_reduce");
            let call = ctx
                .builder
                .ins()
                .call(func_ref, &[arr, init_tag, init_data, closure]);
            let results = ctx.builder.inst_results(call);
            return Some(from_tokvalue(ctx, results[0], results[1], result_ty));
        }
        "tok_array_push" => {
            let arr = compile_expr_as_ptr(ctx, &args[0]);
            let val = compile_expr(ctx, &args[1]).expect("codegen: args[1] produced no value");
            let (tag, data) = to_tokvalue(ctx, val, &args[1].ty);
            let func_ref = ctx.get_runtime_func_ref("tok_array_push");
            let call = ctx.builder.ins().call(func_ref, &[arr, tag, data]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
        "tok_value_to_string" => {
            let val = compile_expr(ctx, &args[0]).expect("codegen: args[0] produced no value");
            let (tag, data) = to_tokvalue(ctx, val, &args[0].ty);
            let func_ref = ctx.get_runtime_func_ref("tok_value_to_string");
            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
        "tok_array_concat" => {
            let a = compile_expr_as_ptr(ctx, &args[0]);
            let b = compile_expr_as_ptr(ctx, &args[1]);
            let func_ref = ctx.get_runtime_func_ref("tok_array_concat");
            let call = ctx.builder.ins().call(func_ref, &[a, b]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
        "tok_map_clone" => {
            let src = compile_expr_as_ptr(ctx, &args[0]);
            let func_ref = ctx.get_runtime_func_ref("tok_map_clone");
            let call = ctx.builder.ins().call(func_ref, &[src]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
        "tok_map_set" if args.len() == 3 => {
            // HIR emits tok_map_set(map, key_str, value) — need to split value into (tag, data)
            let map = compile_expr_as_ptr(ctx, &args[0]);
            let key = compile_expr_as_ptr(ctx, &args[1]);
            let val =
                compile_expr(ctx, &args[2]).expect("codegen: map_set value produced no value");
            let (tag, data) = to_tokvalue(ctx, val, &args[2].ty);
            let func_ref = ctx.get_runtime_func_ref("tok_map_set");
            ctx.builder.ins().call(func_ref, &[map, key, tag, data]);
            return None;
        }
        _ => {}
    }
    // Generic runtime call
    let mut arg_vals = Vec::new();
    for arg in args {
        if let Some(v) = compile_expr(ctx, arg) {
            arg_vals.push(v);
        }
    }
    let func_ref = ctx.get_runtime_func_ref(name);
    let call = ctx.builder.ins().call(func_ref, &arg_vals);
    let results = ctx.builder.inst_results(call);
    if results.is_empty() {
        None
    } else {
        Some(results[0])
    }
}
