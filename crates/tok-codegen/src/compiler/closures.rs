// ─── Closure-related codegen ─────────────────────────────────────────

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value};
use cranelift_module::{Linkage, Module};

use std::collections::HashSet;

use tok_hir::hir::*;
use tok_types::Type;

use super::free_vars::{collect_free_vars, collect_free_vars_expr};
use super::{
    alloc_tokvalue_on_stack, cl_type_or_i64, compile_body, compile_expr, from_tokvalue,
    from_tokvalue_raw_data, to_tokvalue, CapturedVar, FuncCtx, PendingLambda, PTR, TAG_HANDLE,
};

/// Compile a lambda expression: capture analysis, env allocation, closure creation.
pub(crate) fn compile_lambda_expr(
    ctx: &mut FuncCtx,
    params: &[HirParam],
    ret_type: &Type,
    body: &[HirStmt],
) -> Option<Value> {
    let lambda_name = format!("__tok_lambda_{}", ctx.compiler.lambda_counter);
    ctx.compiler.lambda_counter += 1;

    let param_names: HashSet<String> = params.iter().map(|p| p.name.clone()).collect();
    let free_var_names = collect_free_vars(body, &param_names);
    let captures = collect_captures(ctx, &free_var_names);

    let mut sig = ctx.compiler.module.make_signature();
    sig.params.push(AbiParam::new(PTR)); // env_ptr
    for _ in params {
        sig.params.push(AbiParam::new(types::I64)); // tag
        sig.params.push(AbiParam::new(types::I64)); // data
    }
    sig.returns.push(AbiParam::new(types::I64)); // result tag
    sig.returns.push(AbiParam::new(types::I64)); // result data

    let func_id = ctx
        .compiler
        .module
        .declare_function(&lambda_name, Linkage::Local, &sig)
        .expect("codegen: failed to declare lambda");

    let pending_idx = ctx.compiler.pending_lambdas.len();
    ctx.compiler.pending_lambdas.push(PendingLambda {
        name: lambda_name.clone(),
        func_id,
        params: params.to_vec(),
        ret_type: ret_type.clone(),
        body: body.to_vec(),
        captures: captures.clone(),
        specialized_param_types: None,
    });

    let func_ref = ctx
        .compiler
        .module
        .declare_func_in_func(func_id, ctx.builder.func);
    let fn_ptr = ctx.builder.ins().func_addr(PTR, func_ref);

    let env_ptr = alloc_capture_env(ctx, &captures);

    ctx.last_lambda_info = Some((func_id, env_ptr, pending_idx));

    let arity = ctx.builder.ins().iconst(types::I32, params.len() as i64);
    let env_count_val = ctx.builder.ins().iconst(types::I32, captures.len() as i64);
    let alloc_ref = ctx.get_runtime_func_ref("tok_closure_alloc");
    let call = ctx
        .builder
        .ins()
        .call(alloc_ref, &[fn_ptr, env_ptr, arity, env_count_val]);
    Some(ctx.builder.inst_results(call)[0])
}

/// Compile a goroutine spawn expression.
pub(crate) fn compile_go_expr(ctx: &mut FuncCtx, body_expr: &HirExpr) -> Option<Value> {
    let thunk_name = format!("__tok_goroutine_{}", ctx.compiler.lambda_counter);
    ctx.compiler.lambda_counter += 1;

    let empty_locals = HashSet::new();
    let mut free_set = HashSet::new();
    collect_free_vars_expr(body_expr, &empty_locals, &mut free_set, 0);
    let captures = collect_captures(ctx, &free_set);

    let mut sig = ctx.compiler.module.make_signature();
    sig.params.push(AbiParam::new(PTR));
    sig.returns.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I64));

    let func_id = ctx
        .compiler
        .module
        .declare_function(&thunk_name, Linkage::Local, &sig)
        .expect("codegen: failed to declare go thunk");

    ctx.compiler.pending_lambdas.push(PendingLambda {
        name: thunk_name.clone(),
        func_id,
        params: vec![],
        ret_type: body_expr.ty.clone(),
        body: vec![HirStmt::Expr(body_expr.clone())],
        captures: captures.clone(),
        specialized_param_types: None,
    });

    let func_ref = ctx
        .compiler
        .module
        .declare_func_in_func(func_id, ctx.builder.func);
    let fn_ptr = ctx.builder.ins().func_addr(PTR, func_ref);

    let env_ptr = alloc_capture_env(ctx, &captures);

    let go_ref = ctx.get_runtime_func_ref("tok_go");
    let call = ctx.builder.ins().call(go_ref, &[fn_ptr, env_ptr]);
    Some(ctx.builder.inst_results(call)[0])
}

/// Compile a channel receive or handle join expression.
pub(crate) fn compile_receive_expr(
    ctx: &mut FuncCtx,
    chan_expr: &HirExpr,
    result_ty: &Type,
) -> Option<Value> {
    let chan = compile_expr(ctx, chan_expr).expect("codegen: channel expr produced no value");
    match &chan_expr.ty {
        Type::Handle(_) => {
            let func_ref = ctx.get_runtime_func_ref("tok_handle_join");
            let call = ctx.builder.ins().call(func_ref, &[chan]);
            let results = ctx.builder.inst_results(call);
            Some(from_tokvalue(ctx, results[0], results[1], result_ty))
        }
        Type::Any => {
            // Any-typed (e.g. function param): runtime dispatch between handle join and channel recv.
            // Load tag from TokValue to determine if it's a handle (TAG_HANDLE) or channel.
            let tag = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), chan, 0);
            let raw_ptr = from_tokvalue_raw_data(ctx, chan);

            let handle_tag = ctx.builder.ins().iconst(types::I64, TAG_HANDLE);
            let is_handle = ctx.builder.ins().icmp(IntCC::Equal, tag, handle_tag);

            let handle_block = ctx.builder.create_block();
            let chan_block = ctx.builder.create_block();
            let merge = ctx.builder.create_block();
            ctx.builder.append_block_param(merge, types::I64);
            ctx.builder.append_block_param(merge, types::I64);

            ctx.builder
                .ins()
                .brif(is_handle, handle_block, &[], chan_block, &[]);

            // Handle path: tok_handle_join
            ctx.builder.switch_to_block(handle_block);
            ctx.builder.seal_block(handle_block);
            let join_ref = ctx.get_runtime_func_ref("tok_handle_join");
            let join_call = ctx.builder.ins().call(join_ref, &[raw_ptr]);
            let join_results = ctx.builder.inst_results(join_call);
            let h_tag = join_results[0];
            let h_data = join_results[1];
            ctx.builder.ins().jump(merge, &[h_tag, h_data]);

            // Channel path: tok_channel_recv
            ctx.builder.switch_to_block(chan_block);
            ctx.builder.seal_block(chan_block);
            let recv_ref = ctx.get_runtime_func_ref("tok_channel_recv");
            let recv_call = ctx.builder.ins().call(recv_ref, &[raw_ptr]);
            let recv_results = ctx.builder.inst_results(recv_call);
            let c_tag = recv_results[0];
            let c_data = recv_results[1];
            ctx.builder.ins().jump(merge, &[c_tag, c_data]);

            ctx.builder.switch_to_block(merge);
            ctx.builder.seal_block(merge);
            let result_tag = ctx.builder.block_params(merge)[0];
            let result_data = ctx.builder.block_params(merge)[1];
            Some(from_tokvalue(ctx, result_tag, result_data, result_ty))
        }
        _ => {
            let func_ref = ctx.get_runtime_func_ref("tok_channel_recv");
            let call = ctx.builder.ins().call(func_ref, &[chan]);
            let results = ctx.builder.inst_results(call);
            Some(from_tokvalue(ctx, results[0], results[1], result_ty))
        }
    }
}

/// Compile a select expression (non-blocking try of each arm).
pub(crate) fn compile_select_expr(ctx: &mut FuncCtx, arms: &[HirSelectArm]) -> Option<Value> {
    let merge_block = ctx.builder.create_block();
    // Select returns a value (Any) through block params: (tag, data)
    ctx.builder.append_block_param(merge_block, types::I64);
    ctx.builder.append_block_param(merge_block, types::I64);

    let mut default_body: Option<&Vec<HirStmt>> = None;
    let mut channel_arms: Vec<&HirSelectArm> = Vec::new();
    for arm in arms.iter() {
        match arm {
            HirSelectArm::Default(body) => default_body = Some(body),
            _ => channel_arms.push(arm),
        }
    }

    for arm in channel_arms.iter() {
        let next_block = ctx.builder.create_block();
        let body_block = ctx.builder.create_block();

        match arm {
            HirSelectArm::Recv { var, chan, body } => {
                let chan_val =
                    compile_expr(ctx, chan).expect("codegen: channel expr produced no value");
                let ss = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    16,
                    3,
                ));
                let out_ptr = ctx.builder.ins().stack_addr(PTR, ss, 0);
                let try_recv_ref = ctx.get_runtime_func_ref("tok_channel_try_recv");
                let call = ctx.builder.ins().call(try_recv_ref, &[chan_val, out_ptr]);
                let ok = ctx.builder.inst_results(call)[0];
                ctx.builder.ins().brif(ok, body_block, &[], next_block, &[]);

                ctx.builder.switch_to_block(body_block);
                ctx.builder.seal_block(body_block);
                let ct = cl_type_or_i64(&Type::Any);
                let v = ctx.new_var(ct);
                ctx.builder.def_var(v, out_ptr);
                ctx.vars.insert(var.clone(), (v, Type::Any));
                ctx.block_terminated = false;
                let body_val = compile_body(ctx, body, &Type::Any);
                if !ctx.block_terminated {
                    let (tag, data) = select_body_to_tokvalue(ctx, body_val, body);
                    ctx.builder.ins().jump(merge_block, &[tag, data]);
                }
            }
            HirSelectArm::Send { chan, value, body } => {
                let chan_val =
                    compile_expr(ctx, chan).expect("codegen: channel expr produced no value");
                let val = compile_expr(ctx, value).expect("codegen: value expr produced no value");
                let (tag, data) = to_tokvalue(ctx, val, &value.ty);
                let try_send_ref = ctx.get_runtime_func_ref("tok_channel_try_send");
                let call = ctx.builder.ins().call(try_send_ref, &[chan_val, tag, data]);
                let ok = ctx.builder.inst_results(call)[0];
                ctx.builder.ins().brif(ok, body_block, &[], next_block, &[]);

                ctx.builder.switch_to_block(body_block);
                ctx.builder.seal_block(body_block);
                ctx.block_terminated = false;
                let body_val = compile_body(ctx, body, &Type::Any);
                if !ctx.block_terminated {
                    let (tag, data) = select_body_to_tokvalue(ctx, body_val, body);
                    ctx.builder.ins().jump(merge_block, &[tag, data]);
                }
            }
            HirSelectArm::Default(_) => unreachable!(),
        }

        ctx.builder.switch_to_block(next_block);
        ctx.builder.seal_block(next_block);
    }

    if let Some(body) = default_body {
        ctx.block_terminated = false;
        let body_val = compile_body(ctx, body, &Type::Any);
        if !ctx.block_terminated {
            let (tag, data) = select_body_to_tokvalue(ctx, body_val, body);
            ctx.builder.ins().jump(merge_block, &[tag, data]);
        }
    } else if let Some(first_recv) = channel_arms
        .iter()
        .find(|a| matches!(a, HirSelectArm::Recv { .. }))
    {
        if let HirSelectArm::Recv { var, chan, body } = first_recv {
            let chan_val =
                compile_expr(ctx, chan).expect("codegen: channel expr produced no value");
            let recv_ref = ctx.get_runtime_func_ref("tok_channel_recv");
            let call = ctx.builder.ins().call(recv_ref, &[chan_val]);
            let results = ctx.builder.inst_results(call);
            let tag = results[0];
            let data = results[1];
            let val_ptr = alloc_tokvalue_on_stack(ctx, tag, data);
            let ct = cl_type_or_i64(&Type::Any);
            let v = ctx.new_var(ct);
            ctx.builder.def_var(v, val_ptr);
            ctx.vars.insert(var.clone(), (v, Type::Any));
            ctx.block_terminated = false;
            let body_val = compile_body(ctx, body, &Type::Any);
            if !ctx.block_terminated {
                let (tag, data) = select_body_to_tokvalue(ctx, body_val, body);
                ctx.builder.ins().jump(merge_block, &[tag, data]);
            }
        }
    } else {
        // No default, no recv — jump with nil
        let nil_tag = ctx.builder.ins().iconst(types::I64, 0);
        let nil_data = ctx.builder.ins().iconst(types::I64, 0);
        ctx.builder.ins().jump(merge_block, &[nil_tag, nil_data]);
    }

    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
    let result_tag = ctx.builder.block_params(merge_block)[0];
    let result_data = ctx.builder.block_params(merge_block)[1];
    Some(from_tokvalue(ctx, result_tag, result_data, &Type::Any))
}

/// Helper to convert a body's last expression value to (tag, data) for the select merge block.
fn select_body_to_tokvalue(
    ctx: &mut FuncCtx,
    body_val: Option<Value>,
    body: &[HirStmt],
) -> (Value, Value) {
    if let Some(val) = body_val {
        let last_ty = body
            .last()
            .map(|s| match s {
                HirStmt::Expr(e) => e.ty.clone(),
                _ => Type::Nil,
            })
            .unwrap_or(Type::Nil);
        to_tokvalue(ctx, val, &last_ty)
    } else {
        let nil_tag = ctx.builder.ins().iconst(types::I64, 0);
        let nil_data = ctx.builder.ins().iconst(types::I64, 0);
        (nil_tag, nil_data)
    }
}

/// Collect captured variables from a set of free variable names.
pub(crate) fn collect_captures(
    ctx: &FuncCtx,
    free_var_names: &HashSet<String>,
) -> Vec<CapturedVar> {
    let mut captures: Vec<CapturedVar> = Vec::new();
    for name in free_var_names {
        if let Some((_var, var_ty)) = ctx.vars.get(name) {
            captures.push(CapturedVar {
                name: name.clone(),
                ty: var_ty.clone(),
            });
        }
    }
    captures.sort_by(|a, b| a.name.cmp(&b.name));
    captures
}

/// Allocate a capture environment and store captured variables into it.
pub(crate) fn alloc_capture_env(ctx: &mut FuncCtx, captures: &[CapturedVar]) -> Value {
    if captures.is_empty() {
        return ctx.builder.ins().iconst(PTR, 0);
    }
    let count = ctx.builder.ins().iconst(types::I64, captures.len() as i64);
    let alloc_ref = ctx.get_runtime_func_ref("tok_env_alloc");
    let alloc_call = ctx.builder.ins().call(alloc_ref, &[count]);
    let env = ctx.builder.inst_results(alloc_call)[0];

    for (i, cap) in captures.iter().enumerate() {
        let (var, var_ty) = ctx
            .vars
            .get(&cap.name)
            .expect("codegen: captured var not found")
            .clone();
        let val = ctx.builder.use_var(var);
        let (tag, data) = to_tokvalue(ctx, val, &var_ty);
        let offset = (i * 16) as i32;
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), tag, env, offset);
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), data, env, offset + 8);
        let rc_inc_ref = ctx.get_runtime_func_ref("tok_value_rc_inc");
        ctx.builder.ins().call(rc_inc_ref, &[tag, data]);
    }
    env
}
