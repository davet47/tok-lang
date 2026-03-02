// ─── Body / statement compilation ─────────────────────────────────────

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{InstBuilder, MemFlags, Value};
use cranelift_frontend::Variable;

use tok_hir::hir::*;
use tok_types::Type;

use super::{
    alloc_tokvalue_on_stack, cl_type_or_i64, compile_expr, emit_rc_dec, is_heap_type,
    to_tokvalue, unwrap_any_ptr, FuncCtx, KnownClosure, PTR,
};

/// Compile a sequence of statements, returning the value of the last expression.
pub(crate) fn compile_body(ctx: &mut FuncCtx, stmts: &[HirStmt], _expected_type: &Type) -> Option<Value> {
    let mut last_val = None;
    for stmt in stmts {
        last_val = compile_stmt(ctx, stmt);
    }
    last_val
}

/// Compile a single HIR statement. Returns a value if the statement is an expression.
pub(crate) fn compile_stmt(ctx: &mut FuncCtx, stmt: &HirStmt) -> Option<Value> {
    match stmt {
        HirStmt::Assign { name, ty, value } => compile_assign(ctx, name, ty, value),
        HirStmt::FuncDecl { .. } => None, // lifted out during HIR lowering
        HirStmt::IndexAssign {
            target,
            index,
            value,
        } => compile_index_assign(ctx, target, index, value),
        HirStmt::MemberAssign {
            target,
            field,
            value,
        } => compile_member_assign(ctx, target, field, value),
        HirStmt::Expr(expr) => compile_expr(ctx, expr),
        HirStmt::Return(opt_expr) => compile_return(ctx, opt_expr.as_ref()),
        HirStmt::Break => {
            if let Some(&(_, break_block)) = ctx.loop_stack.last() {
                ctx.builder.ins().jump(break_block, &[]);
                switch_to_dead_block(ctx);
            }
            None
        }
        HirStmt::Continue => {
            if let Some(&(continue_block, _)) = ctx.loop_stack.last() {
                ctx.builder.ins().jump(continue_block, &[]);
                switch_to_dead_block(ctx);
            }
            None
        }
        HirStmt::Import(_path) => None, // handled at whole-program level
    }
}

/// Create a dead block after a terminating instruction (break/continue/return).
fn switch_to_dead_block(ctx: &mut FuncCtx) {
    let dead_block = ctx.builder.create_block();
    ctx.builder.switch_to_block(dead_block);
    ctx.builder.seal_block(dead_block);
    ctx.block_terminated = true;
}

/// Compile a variable assignment statement with coercion and RC management.
pub(crate) fn compile_assign(ctx: &mut FuncCtx, name: &str, ty: &Type, value: &HirExpr) -> Option<Value> {
    ctx.last_lambda_info = None;
    let val = compile_expr(ctx, value);
    // If the RHS was a lambda, record it for direct-call optimization
    if let Some((func_id, env_ptr, pending_idx)) = ctx.last_lambda_info.take() {
        ctx.known_closures.insert(
            name.to_string(),
            KnownClosure {
                func_id,
                env_ptr,
                pending_idx,
                specialized: None,
            },
        );
    } else {
        ctx.known_closures.remove(name);
    }
    // Track stdlib module imports for direct-call optimization
    if let HirExprKind::RuntimeCall {
        name: ref call_name,
        ref args,
    } = value.kind
    {
        if call_name == "tok_import" {
            if let Some(arg) = args.first() {
                if let HirExprKind::Str(module_name) = &arg.kind {
                    ctx.stdlib_imports
                        .insert(name.to_string(), module_name.clone());
                }
            }
        }
    } else {
        ctx.stdlib_imports.remove(name);
    }
    if let Some(v) = val {
        if let Some((var, existing_ty)) = ctx.vars.get(name).cloned() {
            let (coerced, effective_ty) = coerce_assign_value(ctx, v, &existing_ty, &value.ty);
            update_var_and_rc_dec(ctx, name, var, &existing_ty, coerced, effective_ty);
        } else {
            let ct = cl_type_or_i64(ty);
            let var = ctx.new_var(ct);
            ctx.builder.def_var(var, v);
            ctx.vars.insert(name.to_string(), (var, ty.clone()));
        }
    }
    None
}

/// Coerce an RHS value to match the existing variable's type, returning (coerced_value, effective_type).
pub(crate) fn coerce_assign_value(
    ctx: &mut FuncCtx,
    v: Value,
    existing_ty: &Type,
    value_ty: &Type,
) -> (Value, Type) {
    let coerced = match (existing_ty, value_ty) {
        // Value is Any but variable is concrete — extract via runtime
        (Type::Int, Type::Any) => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_to_int");
            let (tag, data) = to_tokvalue(ctx, v, &Type::Any);
            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
            ctx.builder.inst_results(call)[0]
        }
        (Type::Float, Type::Any) => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_to_float");
            let (tag, data) = to_tokvalue(ctx, v, &Type::Any);
            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
            ctx.builder.inst_results(call)[0]
        }
        (Type::Bool, Type::Any) => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_to_bool");
            let (tag, data) = to_tokvalue(ctx, v, &Type::Any);
            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
            ctx.builder.inst_results(call)[0]
        }
        (et, Type::Any) if !matches!(et, Type::Any) => {
            let (_, data) = to_tokvalue(ctx, v, &Type::Any);
            data
        }
        // Variable is Any but value is concrete — wrap into TokValue
        (Type::Any, vt) if !matches!(vt, Type::Any | Type::Nil | Type::Never) => {
            let (tag, data) = to_tokvalue(ctx, v, value_ty);
            alloc_tokvalue_on_stack(ctx, tag, data)
        }
        _ => v,
    };
    let effective_ty = match (existing_ty, value_ty) {
        (Type::Int, Type::Any) | (Type::Float, Type::Any) | (Type::Bool, Type::Any) => {
            existing_ty.clone()
        }
        (_, Type::Any) if !matches!(existing_ty, Type::Any) => existing_ty.clone(),
        (Type::Any, vt) if !matches!(vt, Type::Any | Type::Nil | Type::Never) => {
            existing_ty.clone()
        }
        _ => value_ty.clone(),
    };
    (coerced, effective_ty)
}

/// Update a variable binding and emit RC dec for the old value.
pub(crate) fn update_var_and_rc_dec(
    ctx: &mut FuncCtx,
    name: &str,
    var: Variable,
    existing_ty: &Type,
    coerced: Value,
    effective_ty: Type,
) {
    // Save old value for RC dec
    let needs_rc = is_heap_type(existing_ty) || matches!(existing_ty, Type::Any);
    let old_val = if needs_rc {
        Some(ctx.builder.use_var(var))
    } else {
        None
    };

    let new_ct = cl_type_or_i64(&effective_ty);
    let old_ct = cl_type_or_i64(existing_ty);
    let type_changed = std::mem::discriminant(&effective_ty) != std::mem::discriminant(existing_ty);

    if new_ct != old_ct {
        let new_var = ctx.new_var(new_ct);
        ctx.builder.def_var(new_var, coerced);
        ctx.vars.insert(name.to_string(), (new_var, effective_ty));
    } else {
        ctx.builder.def_var(var, coerced);
        if type_changed {
            ctx.vars.insert(name.to_string(), (var, effective_ty));
        }
    }

    // RC: decrement old value now that new value is stored.
    if let Some(old) = old_val {
        if type_changed {
            emit_rc_dec(ctx, old, existing_ty);
        } else if is_heap_type(existing_ty) {
            // Same concrete heap type: only dec if old != new pointer (alias guard)
            let same = ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal,
                old,
                coerced,
            );
            let dec_block = ctx.builder.create_block();
            let cont_block = ctx.builder.create_block();
            ctx.builder
                .ins()
                .brif(same, cont_block, &[], dec_block, &[]);
            ctx.builder.switch_to_block(dec_block);
            ctx.builder.seal_block(dec_block);
            emit_rc_dec(ctx, old, existing_ty);
            ctx.builder.ins().jump(cont_block, &[]);
            ctx.builder.switch_to_block(cont_block);
            ctx.builder.seal_block(cont_block);
        } else if matches!(existing_ty, Type::Any) {
            // Any → Any: old and new are TokValue pointers (different stack slots).
            // Compare the DATA fields (offset 8) — if the underlying heap pointer
            // is the same (e.g., tok_array_push returns the same array), skip RC-dec
            // to avoid premature deallocation.
            let old_data = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), old, 8);
            let new_data = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), coerced, 8);
            let same = ctx.builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal,
                old_data,
                new_data,
            );
            let dec_block = ctx.builder.create_block();
            let cont_block = ctx.builder.create_block();
            ctx.builder
                .ins()
                .brif(same, cont_block, &[], dec_block, &[]);
            ctx.builder.switch_to_block(dec_block);
            ctx.builder.seal_block(dec_block);
            emit_rc_dec(ctx, old, existing_ty);
            ctx.builder.ins().jump(cont_block, &[]);
            ctx.builder.switch_to_block(cont_block);
            ctx.builder.seal_block(cont_block);
        } else {
            emit_rc_dec(ctx, old, existing_ty);
        }
    }
}

/// Compile an index assignment statement (arr[i]=v, map[k]=v).
pub(crate) fn compile_index_assign(
    ctx: &mut FuncCtx,
    target: &HirExpr,
    index: &HirExpr,
    value: &HirExpr,
) -> Option<Value> {
    let target_val = compile_expr(ctx, target).expect("codegen: target expr produced no value");
    let idx_val = compile_expr(ctx, index).expect("codegen: index expr produced no value");
    let val = compile_expr(ctx, value).expect("codegen: value expr produced no value");
    match &target.ty {
        Type::Array(_) => {
            let idx = unwrap_any_ptr(ctx, idx_val, &index.ty);
            let (tag, data) = to_tokvalue(ctx, val, &value.ty);
            let func_ref = ctx.get_runtime_func_ref("tok_array_set");
            ctx.builder
                .ins()
                .call(func_ref, &[target_val, idx, tag, data]);
        }
        Type::Map(_) => {
            let key = unwrap_any_ptr(ctx, idx_val, &index.ty);
            let (tag, data) = to_tokvalue(ctx, val, &value.ty);
            let func_ref = ctx.get_runtime_func_ref("tok_map_set");
            ctx.builder
                .ins()
                .call(func_ref, &[target_val, key, tag, data]);
        }
        Type::Any | Type::Optional(_) | Type::Result(_) => {
            let target_tag = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), target_val, 0);
            let target_data =
                ctx.builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), target_val, 8);
            let (idx_tag, idx_data) = to_tokvalue(ctx, idx_val, &index.ty);
            let (val_tag, val_data) = to_tokvalue(ctx, val, &value.ty);
            let func_ref = ctx.get_runtime_func_ref("tok_value_index_set");
            ctx.builder.ins().call(
                func_ref,
                &[
                    target_tag,
                    target_data,
                    idx_tag,
                    idx_data,
                    val_tag,
                    val_data,
                ],
            );
        }
        _ => {}
    }
    None
}

/// Compile a member assignment statement (map.field = v).
pub(crate) fn compile_member_assign(
    ctx: &mut FuncCtx,
    target: &HirExpr,
    field: &str,
    value: &HirExpr,
) -> Option<Value> {
    let target_val = compile_expr(ctx, target).expect("codegen: target expr produced no value");
    let val = compile_expr(ctx, value).expect("codegen: value expr produced no value");
    let map_ptr = match &target.ty {
        Type::Any | Type::Optional(_) | Type::Result(_) => {
            ctx.builder
                .ins()
                .load(types::I64, MemFlags::trusted(), target_val, 8)
        }
        _ => target_val,
    };
    let (data_id, len) = ctx.compiler.declare_string_data(field);
    let gv = ctx.get_data_ref(data_id);
    let key_ptr = ctx.builder.ins().global_value(PTR, gv);
    let key_len = ctx.builder.ins().iconst(types::I64, len as i64);
    let func_ref = ctx.get_runtime_func_ref("tok_string_alloc");
    let call = ctx.builder.ins().call(func_ref, &[key_ptr, key_len]);
    let key_str = ctx.builder.inst_results(call)[0];
    let (tag, data) = to_tokvalue(ctx, val, &value.ty);
    let set_ref = ctx.get_runtime_func_ref("tok_map_set");
    ctx.builder
        .ins()
        .call(set_ref, &[map_ptr, key_str, tag, data]);
    let free_ref = ctx.get_runtime_func_ref("tok_string_free");
    ctx.builder.ins().call(free_ref, &[key_str]);
    None
}

/// Compile a return statement.
pub(crate) fn compile_return(ctx: &mut FuncCtx, opt_expr: Option<&HirExpr>) -> Option<Value> {
    if let Some(expr) = opt_expr {
        let val = compile_expr(ctx, expr);
        if let Some(v) = val {
            if ctx.is_any_return {
                let (tag, data) = to_tokvalue(ctx, v, &expr.ty);
                ctx.builder.ins().jump(ctx.return_block, &[tag, data]);
            } else {
                ctx.builder.ins().jump(ctx.return_block, &[v]);
            }
        } else if ctx.is_any_return {
            let zero = ctx.builder.ins().iconst(types::I64, 0);
            ctx.builder.ins().jump(ctx.return_block, &[zero, zero]);
        } else {
            ctx.builder.ins().jump(ctx.return_block, &[]);
        }
    } else if ctx.is_any_return {
        let zero = ctx.builder.ins().iconst(types::I64, 0);
        ctx.builder.ins().jump(ctx.return_block, &[zero, zero]);
    } else {
        ctx.builder.ins().jump(ctx.return_block, &[]);
    }
    switch_to_dead_block(ctx);
    None
}
