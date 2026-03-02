// ─── Binary operators ─────────────────────────────────────────────────

use cranelift_codegen::ir::{types, InstBuilder, Value};
use tok_hir::hir::*;
use tok_types::Type;

use super::{compile_expr, from_tokvalue, to_bool, to_tokvalue, FuncCtx};

pub(crate) fn compile_binop(
    ctx: &mut FuncCtx,
    op: HirBinOp,
    left: &HirExpr,
    right: &HirExpr,
    result_ty: &Type,
) -> Option<Value> {
    // Short-circuit for And/Or
    match op {
        HirBinOp::And => return compile_short_circuit_and(ctx, left, right, result_ty),
        HirBinOp::Or => return compile_short_circuit_or(ctx, left, right, result_ty),
        _ => {}
    }

    let lv = compile_expr(ctx, left).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));
    let rv = compile_expr(ctx, right).unwrap_or_else(|| ctx.builder.ins().iconst(types::I64, 0));

    let left_is_any = matches!(&left.ty, Type::Any | Type::Optional(_) | Type::Result(_));
    let right_is_any = matches!(&right.ty, Type::Any | Type::Optional(_) | Type::Result(_));

    // If both sides are Int (and neither is Any)
    if matches!(left.ty, Type::Int)
        && matches!(right.ty, Type::Int)
        && !left_is_any
        && !right_is_any
    {
        return compile_int_binop(ctx, op, lv, rv);
    }

    // If both sides are Float (or one is Float and one is Int), but neither is Any
    if !left_is_any
        && !right_is_any
        && (matches!(left.ty, Type::Float) || matches!(right.ty, Type::Float))
    {
        let lf = if matches!(left.ty, Type::Int) {
            ctx.builder.ins().fcvt_from_sint(types::F64, lv)
        } else {
            lv
        };
        let rf = if matches!(right.ty, Type::Int) {
            ctx.builder.ins().fcvt_from_sint(types::F64, rv)
        } else {
            rv
        };
        return compile_float_binop(ctx, op, lf, rf);
    }

    // String concatenation
    if matches!(left.ty, Type::Str) && matches!(right.ty, Type::Str) && matches!(op, HirBinOp::Add)
    {
        let func_ref = ctx.get_runtime_func_ref("tok_string_concat");
        let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
        return Some(ctx.builder.inst_results(call)[0]);
    }

    // String multiplication: "ha" * 3 or 3 * "ha"
    if matches!(op, HirBinOp::Mul) {
        if matches!(left.ty, Type::Str) && matches!(right.ty, Type::Int) {
            let func_ref = ctx.get_runtime_func_ref("tok_string_repeat");
            let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
        if matches!(left.ty, Type::Int) && matches!(right.ty, Type::Str) {
            let func_ref = ctx.get_runtime_func_ref("tok_string_repeat");
            let call = ctx.builder.ins().call(func_ref, &[rv, lv]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
    }

    // String comparison
    if matches!(left.ty, Type::Str) && matches!(right.ty, Type::Str) {
        match op {
            HirBinOp::Eq => {
                let func_ref = ctx.get_runtime_func_ref("tok_string_eq");
                let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
                return Some(ctx.builder.inst_results(call)[0]);
            }
            HirBinOp::Neq => {
                let func_ref = ctx.get_runtime_func_ref("tok_string_eq");
                let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
                let eq = ctx.builder.inst_results(call)[0];
                let one = ctx.builder.ins().iconst(types::I8, 1);
                return Some(ctx.builder.ins().bxor(eq, one));
            }
            HirBinOp::Lt | HirBinOp::Gt | HirBinOp::LtEq | HirBinOp::GtEq => {
                let func_ref = ctx.get_runtime_func_ref("tok_string_cmp");
                let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
                let cmp = ctx.builder.inst_results(call)[0];
                let zero = ctx.builder.ins().iconst(types::I64, 0);
                let cc = match op {
                    HirBinOp::Lt => cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                    HirBinOp::Gt => cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThan,
                    HirBinOp::LtEq => {
                        cranelift_codegen::ir::condcodes::IntCC::SignedLessThanOrEqual
                    }
                    HirBinOp::GtEq => {
                        cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThanOrEqual
                    }
                    _ => unreachable!(),
                };
                let result = ctx.builder.ins().icmp(cc, cmp, zero);
                return Some(result);
            }
            _ => {}
        }
    }

    // Bool comparisons
    if matches!(left.ty, Type::Bool) && matches!(right.ty, Type::Bool) {
        match op {
            HirBinOp::Eq => {
                let result =
                    ctx.builder
                        .ins()
                        .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, lv, rv);
                return Some(result);
            }
            HirBinOp::Neq => {
                let result = ctx.builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                    lv,
                    rv,
                );
                return Some(result);
            }
            _ => {}
        }
    }

    // Fallback: use runtime value ops for Any types
    let (lt, ld) = to_tokvalue(ctx, lv, &left.ty);
    let (rt, rd) = to_tokvalue(ctx, rv, &right.ty);
    let rt_name = match op {
        HirBinOp::Add => "tok_value_add",
        HirBinOp::Sub => "tok_value_sub",
        HirBinOp::Mul => "tok_value_mul",
        HirBinOp::Div => "tok_value_div",
        HirBinOp::Mod => "tok_value_mod",
        HirBinOp::Eq => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_eq");
            let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
        HirBinOp::Neq => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_eq");
            let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
            let eq = ctx.builder.inst_results(call)[0];
            let one = ctx.builder.ins().iconst(types::I8, 1);
            return Some(ctx.builder.ins().bxor(eq, one));
        }
        HirBinOp::Lt => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_lt");
            let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
            return Some(ctx.builder.inst_results(call)[0]);
        }
        HirBinOp::Gt => {
            let func_ref = ctx.get_runtime_func_ref("tok_value_lt");
            let call = ctx.builder.ins().call(func_ref, &[rt, rd, lt, ld]); // swap
            return Some(ctx.builder.inst_results(call)[0]);
        }
        HirBinOp::LtEq => {
            // a <= b = !(b < a)
            let func_ref = ctx.get_runtime_func_ref("tok_value_lt");
            let call = ctx.builder.ins().call(func_ref, &[rt, rd, lt, ld]);
            let lt_result = ctx.builder.inst_results(call)[0];
            let one = ctx.builder.ins().iconst(types::I8, 1);
            return Some(ctx.builder.ins().bxor(lt_result, one));
        }
        HirBinOp::GtEq => {
            // a >= b = !(a < b)
            let func_ref = ctx.get_runtime_func_ref("tok_value_lt");
            let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
            let lt_result = ctx.builder.inst_results(call)[0];
            let one = ctx.builder.ins().iconst(types::I8, 1);
            return Some(ctx.builder.ins().bxor(lt_result, one));
        }
        HirBinOp::Pow => "tok_value_pow",
        _ => {
            // Bitwise ops — fallback to 0
            return Some(ctx.builder.ins().iconst(types::I64, 0));
        }
    };
    let func_ref = ctx.get_runtime_func_ref(rt_name);
    let call = ctx.builder.ins().call(func_ref, &[lt, ld, rt, rd]);
    let results = ctx.builder.inst_results(call);
    Some(from_tokvalue(ctx, results[0], results[1], result_ty))
}

fn compile_int_binop(ctx: &mut FuncCtx, op: HirBinOp, lv: Value, rv: Value) -> Option<Value> {
    use cranelift_codegen::ir::condcodes::IntCC;
    Some(match op {
        HirBinOp::Add => ctx.builder.ins().iadd(lv, rv),
        HirBinOp::Sub => ctx.builder.ins().isub(lv, rv),
        HirBinOp::Mul => ctx.builder.ins().imul(lv, rv),
        HirBinOp::Div => {
            // Safe sdiv: div-by-zero → 0, i64::MIN / -1 → i64::MIN (wrapping)
            let zero = ctx.builder.ins().iconst(types::I64, 0);
            let is_zero = ctx.builder.ins().icmp(IntCC::Equal, rv, zero);
            let safe_block = ctx.builder.create_block();
            let overflow_check = ctx.builder.create_block();
            let div_block = ctx.builder.create_block();
            let merge = ctx.builder.create_block();
            ctx.builder.append_block_param(merge, types::I64);
            // divisor == 0 → return 0
            ctx.builder
                .ins()
                .brif(is_zero, merge, &[zero], safe_block, &[]);
            // Check i64::MIN / -1
            ctx.builder.switch_to_block(safe_block);
            ctx.builder.seal_block(safe_block);
            let min_val = ctx.builder.ins().iconst(types::I64, i64::MIN);
            let is_min = ctx.builder.ins().icmp(IntCC::Equal, lv, min_val);
            ctx.builder
                .ins()
                .brif(is_min, overflow_check, &[], div_block, &[]);
            ctx.builder.switch_to_block(overflow_check);
            ctx.builder.seal_block(overflow_check);
            let neg1 = ctx.builder.ins().iconst(types::I64, -1i64);
            let is_neg1 = ctx.builder.ins().icmp(IntCC::Equal, rv, neg1);
            ctx.builder
                .ins()
                .brif(is_neg1, merge, &[min_val], div_block, &[]);
            // Normal sdiv
            ctx.builder.switch_to_block(div_block);
            ctx.builder.seal_block(div_block);
            let result = ctx.builder.ins().sdiv(lv, rv);
            ctx.builder.ins().jump(merge, &[result]);
            ctx.builder.switch_to_block(merge);
            ctx.builder.seal_block(merge);
            ctx.builder.block_params(merge)[0]
        }
        HirBinOp::Mod => {
            // Safe srem: div-by-zero → 0, i64::MIN % -1 → 0
            let zero = ctx.builder.ins().iconst(types::I64, 0);
            let is_zero = ctx.builder.ins().icmp(IntCC::Equal, rv, zero);
            let safe_block = ctx.builder.create_block();
            let overflow_check = ctx.builder.create_block();
            let rem_block = ctx.builder.create_block();
            let merge = ctx.builder.create_block();
            ctx.builder.append_block_param(merge, types::I64);
            // divisor == 0 → return 0
            ctx.builder
                .ins()
                .brif(is_zero, merge, &[zero], safe_block, &[]);
            // Check i64::MIN % -1
            ctx.builder.switch_to_block(safe_block);
            ctx.builder.seal_block(safe_block);
            let min_val = ctx.builder.ins().iconst(types::I64, i64::MIN);
            let is_min = ctx.builder.ins().icmp(IntCC::Equal, lv, min_val);
            ctx.builder
                .ins()
                .brif(is_min, overflow_check, &[], rem_block, &[]);
            ctx.builder.switch_to_block(overflow_check);
            ctx.builder.seal_block(overflow_check);
            let neg1 = ctx.builder.ins().iconst(types::I64, -1i64);
            let is_neg1 = ctx.builder.ins().icmp(IntCC::Equal, rv, neg1);
            ctx.builder
                .ins()
                .brif(is_neg1, merge, &[zero], rem_block, &[]);
            // Normal srem
            ctx.builder.switch_to_block(rem_block);
            ctx.builder.seal_block(rem_block);
            let result = ctx.builder.ins().srem(lv, rv);
            ctx.builder.ins().jump(merge, &[result]);
            ctx.builder.switch_to_block(merge);
            ctx.builder.seal_block(merge);
            ctx.builder.block_params(merge)[0]
        }
        HirBinOp::Pow => {
            let func_ref = ctx.get_runtime_func_ref("tok_pow_int");
            let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
            ctx.builder.inst_results(call)[0]
        }
        HirBinOp::Eq => ctx.builder.ins().icmp(IntCC::Equal, lv, rv),
        HirBinOp::Neq => ctx.builder.ins().icmp(IntCC::NotEqual, lv, rv),
        HirBinOp::Lt => ctx.builder.ins().icmp(IntCC::SignedLessThan, lv, rv),
        HirBinOp::Gt => ctx.builder.ins().icmp(IntCC::SignedGreaterThan, lv, rv),
        HirBinOp::LtEq => ctx.builder.ins().icmp(IntCC::SignedLessThanOrEqual, lv, rv),
        HirBinOp::GtEq => ctx
            .builder
            .ins()
            .icmp(IntCC::SignedGreaterThanOrEqual, lv, rv),
        HirBinOp::BitAnd => ctx.builder.ins().band(lv, rv),
        HirBinOp::BitOr => ctx.builder.ins().bor(lv, rv),
        HirBinOp::BitXor => ctx.builder.ins().bxor(lv, rv),
        HirBinOp::Shr => ctx.builder.ins().sshr(lv, rv),
        HirBinOp::And | HirBinOp::Or => unreachable!("handled by short-circuit"),
    })
}

fn compile_float_binop(ctx: &mut FuncCtx, op: HirBinOp, lv: Value, rv: Value) -> Option<Value> {
    use cranelift_codegen::ir::condcodes::FloatCC;
    Some(match op {
        HirBinOp::Add => ctx.builder.ins().fadd(lv, rv),
        HirBinOp::Sub => ctx.builder.ins().fsub(lv, rv),
        HirBinOp::Mul => ctx.builder.ins().fmul(lv, rv),
        HirBinOp::Div => ctx.builder.ins().fdiv(lv, rv),
        HirBinOp::Mod => {
            // fmod: a - floor(a/b) * b
            let div = ctx.builder.ins().fdiv(lv, rv);
            let floored = ctx.builder.ins().floor(div);
            let prod = ctx.builder.ins().fmul(floored, rv);
            ctx.builder.ins().fsub(lv, prod)
        }
        HirBinOp::Pow => {
            let func_ref = ctx.get_runtime_func_ref("tok_pow_f64");
            let call = ctx.builder.ins().call(func_ref, &[lv, rv]);
            ctx.builder.inst_results(call)[0]
        }
        HirBinOp::Eq => ctx.builder.ins().fcmp(FloatCC::Equal, lv, rv),
        HirBinOp::Neq => ctx.builder.ins().fcmp(FloatCC::NotEqual, lv, rv),
        HirBinOp::Lt => ctx.builder.ins().fcmp(FloatCC::LessThan, lv, rv),
        HirBinOp::Gt => ctx.builder.ins().fcmp(FloatCC::GreaterThan, lv, rv),
        HirBinOp::LtEq => ctx.builder.ins().fcmp(FloatCC::LessThanOrEqual, lv, rv),
        HirBinOp::GtEq => ctx.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lv, rv),
        _ => lv, // bitwise ops on float don't make sense
    })
}

// ─── Short-circuit logic ──────────────────────────────────────────────

fn compile_short_circuit_and(
    ctx: &mut FuncCtx,
    left: &HirExpr,
    right: &HirExpr,
    _result_ty: &Type,
) -> Option<Value> {
    let lv = compile_expr(ctx, left).expect("codegen: left operand produced no value");
    let then_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.append_block_param(merge_block, types::I8);

    // If left is falsy, short-circuit to false
    let cond = to_bool(ctx, lv, &left.ty);
    let false_val = ctx.builder.ins().iconst(types::I8, 0);
    ctx.builder
        .ins()
        .brif(cond, then_block, &[], merge_block, &[false_val]);

    ctx.builder.switch_to_block(then_block);
    ctx.builder.seal_block(then_block);
    let rv = compile_expr(ctx, right).expect("codegen: right operand produced no value");
    let right_bool = to_bool(ctx, rv, &right.ty);
    ctx.builder.ins().jump(merge_block, &[right_bool]);

    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
    Some(ctx.builder.block_params(merge_block)[0])
}

fn compile_short_circuit_or(
    ctx: &mut FuncCtx,
    left: &HirExpr,
    right: &HirExpr,
    _result_ty: &Type,
) -> Option<Value> {
    let lv = compile_expr(ctx, left).expect("codegen: left operand produced no value");
    let else_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.append_block_param(merge_block, types::I8);

    let cond = to_bool(ctx, lv, &left.ty);
    let true_val = ctx.builder.ins().iconst(types::I8, 1);
    ctx.builder
        .ins()
        .brif(cond, merge_block, &[true_val], else_block, &[]);

    ctx.builder.switch_to_block(else_block);
    ctx.builder.seal_block(else_block);
    let rv = compile_expr(ctx, right).expect("codegen: right operand produced no value");
    let right_bool = to_bool(ctx, rv, &right.ty);
    ctx.builder.ins().jump(merge_block, &[right_bool]);

    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
    Some(ctx.builder.block_params(merge_block)[0])
}
