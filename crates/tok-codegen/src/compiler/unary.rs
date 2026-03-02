// ─── Unary operators ──────────────────────────────────────────────────

use cranelift_codegen::ir::{types, InstBuilder, Value};
use tok_types::Type;
use tok_hir::hir::*;

use super::{FuncCtx, compile_expr, to_tokvalue, from_tokvalue};

pub(crate) fn compile_unaryop(
    ctx: &mut FuncCtx,
    op: HirUnaryOp,
    operand: &HirExpr,
    _result_ty: &Type,
) -> Option<Value> {
    let val = compile_expr(ctx, operand).expect("codegen: operand expr produced no value");
    match op {
        HirUnaryOp::Neg => {
            if matches!(operand.ty, Type::Int) {
                Some(ctx.builder.ins().ineg(val))
            } else if matches!(operand.ty, Type::Float) {
                Some(ctx.builder.ins().fneg(val))
            } else {
                let (tag, data) = to_tokvalue(ctx, val, &operand.ty);
                let func_ref = ctx.get_runtime_func_ref("tok_value_negate");
                let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                let results = ctx.builder.inst_results(call);
                Some(from_tokvalue(ctx, results[0], results[1], &operand.ty))
            }
        }
        HirUnaryOp::Not => {
            if matches!(operand.ty, Type::Bool) {
                let one = ctx.builder.ins().iconst(types::I8, 1);
                Some(ctx.builder.ins().bxor(val, one))
            } else {
                let (tag, data) = to_tokvalue(ctx, val, &operand.ty);
                let func_ref = ctx.get_runtime_func_ref("tok_value_not");
                let call = ctx.builder.ins().call(func_ref, &[tag, data]);
                Some(ctx.builder.inst_results(call)[0])
            }
        }
    }
}
