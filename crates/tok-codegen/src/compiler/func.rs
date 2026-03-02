// ─── Function-level compilation ──────────────────────────────────────
//
// Free-standing functions that compile named functions, lambdas, the main
// function, and the C entry point into Cranelift IR.

use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, MemFlags, UserFuncName, Value};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{Linkage, Module};

use std::collections::{HashMap, HashSet};

use tok_hir::hir::*;
use tok_types::Type;

use super::{
    alloc_tokvalue_on_stack, cl_type, cl_type_or_i64, coerce_value, compile_body, emit_rc_dec,
    is_heap_type, is_self_tail_recursive, to_tokvalue, zero_value, CapturedVar, Compiler, FuncCtx,
    PendingLambda, PTR,
};

/// Compile a named function.
pub(crate) fn compile_function(
    compiler: &mut Compiler,
    name: &str,
    params: &[HirParam],
    ret_type: &Type,
    body: &[HirStmt],
) {
    let func_id = compiler.declare_tok_func(name, params, ret_type);
    let sig = compiler
        .module
        .declarations()
        .get_function_decl(func_id)
        .signature
        .clone();

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, func_id.as_u32());

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    let return_block = builder.create_block();

    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Set up return
    let is_any_return = matches!(ret_type, Type::Any);
    let return_var = if is_any_return {
        // Any return: return_block takes two params (tag, data)
        builder.append_block_param(return_block, types::I64);
        builder.append_block_param(return_block, types::I64);
        // Use a dummy variable index — won't be used directly
        let rv = Variable::new(0);
        builder.declare_var(rv, types::I64);
        let zero = zero_value(&mut builder, types::I64);
        builder.def_var(rv, zero);
        Some(rv)
    } else if let Some(ct) = cl_type(ret_type) {
        let rv = Variable::new(0);
        builder.declare_var(rv, ct);
        // Initialize return var to zero
        let zero = zero_value(&mut builder, ct);
        builder.def_var(rv, zero);
        // Add return block param
        builder.append_block_param(return_block, ct);
        Some(rv)
    } else {
        None
    };

    let next_var_start = if return_var.is_some() { 1 } else { 0 };

    let mut func_ctx = FuncCtx {
        compiler,
        builder,
        vars: HashMap::new(),
        next_var: next_var_start,
        return_block,
        return_var,
        loop_stack: Vec::new(),
        block_terminated: false,
        is_any_return,
        ret_type: ret_type.clone(),
        known_closures: HashMap::new(),
        stdlib_imports: HashMap::new(),
        last_lambda_info: None,
        param_names: HashSet::new(),
        closure_sig_cache: HashMap::new(),
        tco_func_name: None,
        tco_loop_header: None,
        tco_param_vars: Vec::new(),
    };

    // Check for tail-call optimization opportunity
    let use_tco = is_self_tail_recursive(body, name);

    // Collect initial param values from entry block params
    let mut entry_param_vals = Vec::new();
    {
        let mut bpi = 0;
        for param in params.iter() {
            if matches!(param.ty, Type::Any) {
                let tag_val = func_ctx.builder.block_params(entry_block)[bpi];
                let data_val = func_ctx.builder.block_params(entry_block)[bpi + 1];
                entry_param_vals.push(tag_val);
                entry_param_vals.push(data_val);
                bpi += 2;
            } else if cl_type(&param.ty).is_some() {
                let val = func_ctx.builder.block_params(entry_block)[bpi];
                entry_param_vals.push(val);
                bpi += 1;
            }
        }
    }

    // If TCO, create loop header block and jump from entry
    let body_block = if use_tco {
        let loop_header = func_ctx.builder.create_block();
        // Add block params matching the function's Cranelift params
        for param in params.iter() {
            if matches!(param.ty, Type::Any) {
                func_ctx.builder.append_block_param(loop_header, types::I64); // tag
                func_ctx.builder.append_block_param(loop_header, types::I64); // data
            } else if let Some(ct) = cl_type(&param.ty) {
                func_ctx.builder.append_block_param(loop_header, ct);
            }
        }
        // Jump from entry to loop header with initial values
        func_ctx.builder.ins().jump(loop_header, &entry_param_vals);
        func_ctx.builder.switch_to_block(loop_header);
        // Don't seal loop_header yet — back-edges will be added by tail calls
        func_ctx.tco_func_name = Some(name.to_string());
        func_ctx.tco_loop_header = Some(loop_header);
        loop_header
    } else {
        entry_block
    };

    // Define parameters as variables
    // For Any params, each takes two block params (tag, data); we store as stack TokValue.
    let mut block_param_idx = 0;
    for param in params.iter() {
        func_ctx.param_names.insert(param.name.clone());
        if matches!(param.ty, Type::Any) {
            // Any param: two block params (tag, data), store as stack TokValue
            let tag_val = func_ctx.builder.block_params(body_block)[block_param_idx];
            let data_val = func_ctx.builder.block_params(body_block)[block_param_idx + 1];
            block_param_idx += 2;
            // Create stack slot and store
            let addr = alloc_tokvalue_on_stack(&mut func_ctx, tag_val, data_val);
            let var = func_ctx.new_var(PTR);
            func_ctx.builder.def_var(var, addr);
            func_ctx
                .vars
                .insert(param.name.clone(), (var, param.ty.clone()));
            func_ctx.tco_param_vars.push(var);
        } else if let Some(ct) = cl_type(&param.ty) {
            let var = func_ctx.new_var(ct);
            let param_val = func_ctx.builder.block_params(body_block)[block_param_idx];
            block_param_idx += 1;
            func_ctx.builder.def_var(var, param_val);
            func_ctx
                .vars
                .insert(param.name.clone(), (var, param.ty.clone()));
            func_ctx.tco_param_vars.push(var);
        }
    }

    // Compile body
    let last_val = compile_body(&mut func_ctx, body, ret_type);

    // Seal TCO loop header now that all back-edges have been added
    if let Some(loop_header) = func_ctx.tco_loop_header {
        func_ctx.builder.seal_block(loop_header);
    }

    // Jump to return block with value, but only if the current block isn't
    // already terminated (e.g., by a Return statement that already jumped).
    if !func_ctx.block_terminated {
        if is_any_return {
            if let Some(val) = last_val {
                // Determine the actual type of the last expression from the HIR,
                // so we use the correct type for to_tokvalue (not just ret_type=Any).
                let last_expr_ty = body
                    .last()
                    .and_then(|s| match s {
                        HirStmt::Expr(e) => Some(e.ty.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| ret_type.clone());
                let (tag, data) = to_tokvalue(&mut func_ctx, val, &last_expr_ty);
                func_ctx.builder.ins().jump(return_block, &[tag, data]);
            } else {
                let zero = func_ctx.builder.ins().iconst(types::I64, 0);
                func_ctx.builder.ins().jump(return_block, &[zero, zero]);
            }
        } else if let Some(_rv) = return_var {
            if let Some(val) = last_val {
                func_ctx.builder.ins().jump(return_block, &[val]);
            } else {
                let default = zero_value(&mut func_ctx.builder, cl_type_or_i64(ret_type));
                func_ctx.builder.ins().jump(return_block, &[default]);
            }
        } else {
            func_ctx.builder.ins().jump(return_block, &[]);
        }
    }

    // Return block
    func_ctx.builder.switch_to_block(return_block);
    func_ctx.builder.seal_block(return_block);

    // RC cleanup: dec all heap-typed locals (skip params — caller owns them)
    let heap_locals: Vec<(String, Variable, Type)> = func_ctx
        .vars
        .iter()
        .filter(|(name, (_, ty))| {
            !func_ctx.param_names.contains(name.as_str())
                && (is_heap_type(ty) || matches!(ty, Type::Any))
        })
        .map(|(name, (var, ty))| (name.clone(), *var, ty.clone()))
        .collect();

    if is_any_return {
        let tag_ret = func_ctx.builder.block_params(return_block)[0];
        let data_ret = func_ctx.builder.block_params(return_block)[1];

        // For Any return, data_ret might be a heap pointer.
        // Use aliasing guard: compare each local against data_ret, skip if same.
        for (_, var, ty) in &heap_locals {
            let v = func_ctx.builder.use_var(*var);
            // For Any-typed locals, the pointer is inside the TokValue stack slot;
            // for heap-typed locals, v is the pointer directly.
            let ptr = if matches!(ty, Type::Any) {
                // Load the data field (offset +8) from the stack TokValue
                func_ctx
                    .builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), v, 8)
            } else {
                v
            };
            let same = func_ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal,
                ptr,
                data_ret,
            );
            let dec_block = func_ctx.builder.create_block();
            let cont_block = func_ctx.builder.create_block();
            func_ctx
                .builder
                .ins()
                .brif(same, cont_block, &[], dec_block, &[]);
            func_ctx.builder.switch_to_block(dec_block);
            func_ctx.builder.seal_block(dec_block);
            emit_rc_dec(&mut func_ctx, v, ty);
            func_ctx.builder.ins().jump(cont_block, &[]);
            func_ctx.builder.switch_to_block(cont_block);
            func_ctx.builder.seal_block(cont_block);
        }

        func_ctx.builder.ins().return_(&[tag_ret, data_ret]);
    } else if let Some(_rv) = return_var {
        let ret_val = func_ctx.builder.block_params(return_block)[0];

        if is_heap_type(ret_type) {
            // Return is a heap pointer — use aliasing guard per local
            for (_, var, ty) in &heap_locals {
                let v = func_ctx.builder.use_var(*var);
                let same = func_ctx.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::Equal,
                    v,
                    ret_val,
                );
                let dec_block = func_ctx.builder.create_block();
                let cont_block = func_ctx.builder.create_block();
                func_ctx
                    .builder
                    .ins()
                    .brif(same, cont_block, &[], dec_block, &[]);
                func_ctx.builder.switch_to_block(dec_block);
                func_ctx.builder.seal_block(dec_block);
                emit_rc_dec(&mut func_ctx, v, ty);
                func_ctx.builder.ins().jump(cont_block, &[]);
                func_ctx.builder.switch_to_block(cont_block);
                func_ctx.builder.seal_block(cont_block);
            }
        } else {
            // Return is scalar — safe to unconditionally dec all heap locals
            for (_, var, ty) in &heap_locals {
                let v = func_ctx.builder.use_var(*var);
                emit_rc_dec(&mut func_ctx, v, ty);
            }
        }

        func_ctx.builder.ins().return_(&[ret_val]);
    } else {
        // Void return: unconditionally dec all heap locals
        for (_, var, ty) in &heap_locals {
            let v = func_ctx.builder.use_var(*var);
            emit_rc_dec(&mut func_ctx, v, ty);
        }
        func_ctx.builder.ins().return_(&[]);
    }

    func_ctx.builder.finalize();

    let mut ctx = Context::for_function(func);
    compiler
        .module
        .define_function(func_id, &mut ctx)
        .expect("codegen: failed to define function");
}

/// Load captured variables from an environment pointer into the function context.
///
/// Each capture occupies 16 bytes (tag @ offset 0, data @ offset 8) in the env buffer.
/// When `specialized` is true, concrete types (Int, Float, Bool) are extracted as native
/// values; other types fall back to Any (TokValue on stack). When `specialized` is false,
/// all captures are loaded as Any regardless of their declared type.
pub(crate) fn load_captures_from_env(
    ctx: &mut FuncCtx,
    captures: &[CapturedVar],
    env_ptr: Value,
    specialized: bool,
) {
    for (i, cap) in captures.iter().enumerate() {
        let offset = (i * 16) as i32;
        if specialized {
            match &cap.ty {
                Type::Int => {
                    let data = ctx.builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        env_ptr,
                        offset + 8,
                    );
                    let var = ctx.new_var(types::I64);
                    ctx.builder.def_var(var, data);
                    ctx.vars.insert(cap.name.clone(), (var, Type::Int));
                }
                Type::Float => {
                    let data = ctx.builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        env_ptr,
                        offset + 8,
                    );
                    let fval = ctx.builder.ins().bitcast(types::F64, MemFlags::new(), data);
                    let var = ctx.new_var(types::F64);
                    ctx.builder.def_var(var, fval);
                    ctx.vars.insert(cap.name.clone(), (var, Type::Float));
                }
                Type::Bool => {
                    let data = ctx.builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        env_ptr,
                        offset + 8,
                    );
                    let bval = ctx.builder.ins().ireduce(types::I8, data);
                    let var = ctx.new_var(types::I8);
                    ctx.builder.def_var(var, bval);
                    ctx.vars.insert(cap.name.clone(), (var, Type::Bool));
                }
                _ => {
                    // Non-primitive type — load as Any (TokValue on stack)
                    let tag =
                        ctx.builder
                            .ins()
                            .load(types::I64, MemFlags::trusted(), env_ptr, offset);
                    let data = ctx.builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        env_ptr,
                        offset + 8,
                    );
                    let addr = alloc_tokvalue_on_stack(ctx, tag, data);
                    let var = ctx.new_var(PTR);
                    ctx.builder.def_var(var, addr);
                    ctx.vars.insert(cap.name.clone(), (var, Type::Any));
                }
            }
        } else {
            // Generic path: load all captures as Any
            let tag = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), env_ptr, offset);
            let data = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), env_ptr, offset + 8);
            let addr = alloc_tokvalue_on_stack(ctx, tag, data);
            let var = ctx.new_var(PTR);
            ctx.builder.def_var(var, addr);
            ctx.vars.insert(cap.name.clone(), (var, Type::Any));
        }
    }
}

/// Compile a deferred lambda body into its own Cranelift function.
///
/// Lambda calling convention: (env_ptr: PTR, arg0_tag: I64, arg0_data: I64, ...) -> (I64, I64)
/// All params are passed as TokValue (tag, data) pairs, return is a TokValue pair.
pub(crate) fn compile_lambda_body(compiler: &mut Compiler, lambda: &PendingLambda) {
    let sig = compiler
        .module
        .declarations()
        .get_function_decl(lambda.func_id)
        .signature
        .clone();

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, lambda.func_id.as_u32());

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    let return_block = builder.create_block();

    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Return block takes (tag: I64, data: I64)
    builder.append_block_param(return_block, types::I64);
    builder.append_block_param(return_block, types::I64);

    // Use a dummy return variable (won't be used directly since we use block params)
    let rv = Variable::new(0);
    builder.declare_var(rv, types::I64);
    let zero_val = builder.ins().iconst(types::I64, 0);
    builder.def_var(rv, zero_val);

    let mut func_ctx = FuncCtx {
        compiler,
        builder,
        vars: HashMap::new(),
        next_var: 1, // 0 is used by return var
        return_block,
        return_var: Some(rv),
        loop_stack: Vec::new(),
        block_terminated: false,
        is_any_return: true, // lambdas always return (tag, data)
        ret_type: lambda.ret_type.clone(),
        known_closures: HashMap::new(),
        stdlib_imports: HashMap::new(),
        last_lambda_info: None,
        param_names: HashSet::new(),
        closure_sig_cache: HashMap::new(),
        tco_func_name: None,
        tco_loop_header: None,
        tco_param_vars: Vec::new(),
    };

    // Bind parameters: first block param is env_ptr, then (tag, data) pairs
    let env_ptr_val = func_ctx.builder.block_params(entry_block)[0];
    let mut block_param_idx = 1; // skip env_ptr at index 0
    for param in &lambda.params {
        let tag_val = func_ctx.builder.block_params(entry_block)[block_param_idx];
        let data_val = func_ctx.builder.block_params(entry_block)[block_param_idx + 1];
        block_param_idx += 2;
        // RC-increment parameter values so the lambda owns its own reference.
        // Without this, reassigning a parameter (e.g. `a<<=val`) would RC-dec
        // the caller's reference to 0, freeing the value while the caller
        // still holds a pointer to it.
        let rc_inc_ref = func_ctx.get_runtime_func_ref("tok_value_rc_inc");
        func_ctx
            .builder
            .ins()
            .call(rc_inc_ref, &[tag_val, data_val]);
        // Store as stack-allocated TokValue and bind to param name as Any
        let addr = alloc_tokvalue_on_stack(&mut func_ctx, tag_val, data_val);
        let var = func_ctx.new_var(PTR);
        func_ctx.builder.def_var(var, addr);
        func_ctx.vars.insert(param.name.clone(), (var, Type::Any));
    }

    // Load captured variables from env_ptr (all as Any for generic lambda)
    load_captures_from_env(&mut func_ctx, &lambda.captures, env_ptr_val, false);

    // Compile body
    let last_val = compile_body(&mut func_ctx, &lambda.body, &lambda.ret_type);

    // Jump to return block with value
    if !func_ctx.block_terminated {
        if let Some(val) = last_val {
            // Use the actual type of the last expression, not just ret_type
            let last_expr_ty = lambda
                .body
                .last()
                .and_then(|s| match s {
                    HirStmt::Expr(e) => Some(e.ty.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| lambda.ret_type.clone());
            let (tag, data) = to_tokvalue(&mut func_ctx, val, &last_expr_ty);
            func_ctx.builder.ins().jump(return_block, &[tag, data]);
        } else {
            let zero = func_ctx.builder.ins().iconst(types::I64, 0);
            func_ctx.builder.ins().jump(return_block, &[zero, zero]);
        }
    }

    // Return block
    func_ctx.builder.switch_to_block(return_block);
    func_ctx.builder.seal_block(return_block);
    let tag_ret = func_ctx.builder.block_params(return_block)[0];
    let data_ret = func_ctx.builder.block_params(return_block)[1];

    // Note: We intentionally do NOT RC-dec locals here. The RC-inc at lambda
    // entry for parameters prevents use-after-free when params are reassigned.
    // The slight over-count (leak) is acceptable for short-lived programs;
    // a full RC cleanup would require tracking which vars are returned to
    // avoid double-free of the return value.

    func_ctx.builder.ins().return_(&[tag_ret, data_ret]);

    func_ctx.builder.finalize();

    let mut ctx = Context::for_function(func);
    compiler
        .module
        .define_function(lambda.func_id, &mut ctx)
        .expect("codegen: failed to define lambda function");
}

/// Compile a specialized lambda body with native-typed calling convention.
///
/// Specialized calling convention: (env_ptr: PTR, arg0: T0, arg1: T1, ...) -> RetT
/// Params are native types, no boxing/unboxing.
pub(crate) fn compile_specialized_lambda_body(compiler: &mut Compiler, lambda: &PendingLambda) {
    let spec_types = lambda
        .specialized_param_types
        .as_ref()
        .expect("codegen: specialized lambda missing param types");
    let sig = compiler
        .module
        .declarations()
        .get_function_decl(lambda.func_id)
        .signature
        .clone();

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, lambda.func_id.as_u32());

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    let return_block = builder.create_block();

    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Return block takes a single native-typed value
    let cl_ret = cl_type_or_i64(&lambda.ret_type);
    builder.append_block_param(return_block, cl_ret);

    let rv = Variable::new(0);
    builder.declare_var(rv, cl_ret);
    let zero_val = zero_value(&mut builder, cl_ret);
    builder.def_var(rv, zero_val);

    let mut func_ctx = FuncCtx {
        compiler,
        builder,
        vars: HashMap::new(),
        next_var: 1,
        return_block,
        return_var: Some(rv),
        loop_stack: Vec::new(),
        block_terminated: false,
        is_any_return: false, // specialized: single native return
        ret_type: lambda.ret_type.clone(),
        known_closures: HashMap::new(),
        stdlib_imports: HashMap::new(),
        last_lambda_info: None,
        param_names: HashSet::new(),
        closure_sig_cache: HashMap::new(),
        tco_func_name: None,
        tco_loop_header: None,
        tco_param_vars: Vec::new(),
    };

    // Bind parameters: first block param is env_ptr, then one native value per param
    let env_ptr_val = func_ctx.builder.block_params(entry_block)[0];
    for (i, (param, param_ty)) in lambda.params.iter().zip(spec_types.iter()).enumerate() {
        let val = func_ctx.builder.block_params(entry_block)[1 + i];
        let ct = cl_type_or_i64(param_ty);
        let var = func_ctx.new_var(ct);
        func_ctx.builder.def_var(var, val);
        func_ctx
            .vars
            .insert(param.name.clone(), (var, param_ty.clone()));
    }

    // Load captured variables from env_ptr (with type-specific extraction)
    load_captures_from_env(&mut func_ctx, &lambda.captures, env_ptr_val, true);

    // Compile body
    let last_val = compile_body(&mut func_ctx, &lambda.body, &lambda.ret_type);

    // Jump to return block with native value
    if !func_ctx.block_terminated {
        if let Some(val) = last_val {
            let last_expr_ty = lambda
                .body
                .last()
                .and_then(|s| match s {
                    HirStmt::Expr(e) => Some(e.ty.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| lambda.ret_type.clone());
            let coerced = coerce_value(&mut func_ctx, val, &last_expr_ty, &lambda.ret_type);
            func_ctx.builder.ins().jump(return_block, &[coerced]);
        } else {
            let zero = func_ctx.builder.ins().iconst(cl_ret, 0);
            func_ctx.builder.ins().jump(return_block, &[zero]);
        }
    }

    // Return block
    func_ctx.builder.switch_to_block(return_block);
    func_ctx.builder.seal_block(return_block);
    let ret_val = func_ctx.builder.block_params(return_block)[0];
    func_ctx.builder.ins().return_(&[ret_val]);

    func_ctx.builder.finalize();

    let mut ctx = Context::for_function(func);
    compiler
        .module
        .define_function(lambda.func_id, &mut ctx)
        .expect("codegen: failed to define specialized lambda");
}

/// Compile top-level statements into `_tok_main`.
pub(crate) fn compile_main(compiler: &mut Compiler, stmts: &[HirStmt]) {
    let sig = compiler.module.make_signature();
    // _tok_main returns void
    let func_id = compiler
        .module
        .declare_function("_tok_main", Linkage::Export, &sig)
        .expect("codegen: failed to declare _tok_main");

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, func_id.as_u32());

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    let return_block = builder.create_block();

    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let mut func_ctx = FuncCtx {
        compiler,
        builder,
        vars: HashMap::new(),
        next_var: 0,
        return_block,
        return_var: None,
        loop_stack: Vec::new(),
        block_terminated: false,
        is_any_return: false,
        ret_type: Type::Nil,
        known_closures: HashMap::new(),
        stdlib_imports: HashMap::new(),
        last_lambda_info: None,
        param_names: HashSet::new(),
        closure_sig_cache: HashMap::new(),
        tco_func_name: None,
        tco_loop_header: None,
        tco_param_vars: Vec::new(),
    };

    compile_body(&mut func_ctx, stmts, &Type::Nil);

    if !func_ctx.block_terminated {
        func_ctx.builder.ins().jump(return_block, &[]);
    }
    func_ctx.builder.switch_to_block(return_block);
    func_ctx.builder.seal_block(return_block);

    // RC cleanup: dec all heap-typed locals before exit (main has no params)
    let heap_vars: Vec<(Variable, Type)> = func_ctx
        .vars
        .values()
        .filter(|(_, ty)| is_heap_type(ty) || matches!(ty, Type::Any))
        .map(|(var, ty)| (*var, ty.clone()))
        .collect();
    for (var, ty) in &heap_vars {
        let v = func_ctx.builder.use_var(*var);
        emit_rc_dec(&mut func_ctx, v, ty);
    }

    func_ctx.builder.ins().return_(&[]);
    func_ctx.builder.finalize();

    let mut ctx = Context::for_function(func);
    if std::env::var("CLIF_DUMP").is_ok() {
        eprintln!("=== _tok_main IR ===\n{}", ctx.func.display());
    }
    compiler
        .module
        .define_function(func_id, &mut ctx)
        .expect("codegen: failed to define function");
}

/// Compile the C `main` entry point that calls `_tok_main`.
pub(crate) fn compile_entry(compiler: &mut Compiler) {
    let mut sig = compiler.module.make_signature();
    sig.params.push(AbiParam::new(types::I32)); // argc
    sig.params.push(AbiParam::new(PTR)); // argv
    sig.returns.push(AbiParam::new(types::I32)); // exit code

    let func_id = compiler
        .module
        .declare_function("main", Linkage::Export, &sig)
        .expect("codegen: failed to declare main");

    let mut func = Function::new();
    func.signature = sig;
    func.name = UserFuncName::user(0, func_id.as_u32());

    // Declare _tok_main reference before creating builder (avoids borrow conflict)
    let tok_main_sig = compiler.module.make_signature();
    let tok_main_id = compiler
        .module
        .declare_function("_tok_main", Linkage::Export, &tok_main_sig)
        .expect("codegen: failed to declare _tok_main reference");
    let tok_main_ref = compiler.module.declare_func_in_func(tok_main_id, &mut func);

    let mut func_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_builder_ctx);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Call _tok_main
    builder.ins().call(tok_main_ref, &[]);

    // Return 0
    let zero = builder.ins().iconst(types::I32, 0);
    builder.ins().return_(&[zero]);

    builder.finalize();

    let mut ctx = Context::for_function(func);
    compiler
        .module
        .define_function(func_id, &mut ctx)
        .expect("codegen: failed to define function");
}
