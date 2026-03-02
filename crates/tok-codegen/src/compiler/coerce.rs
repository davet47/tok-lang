// ─── Type coercion system ──────────────────────────────────────────────
//
// The codegen type coercion system converts values between Tok types and the
// runtime's TokValue representation. There are 5 entry points, used in order:
//
// 1. `to_tokvalue(ctx, val, ty)` — Pack a typed value into (tag, data) pair.
//    Use when calling runtime functions that expect TokValue args.
//
// 2. `from_tokvalue(ctx, tag, data, ty)` — Unpack (tag, data) to a typed value.
//    Use when receiving results from runtime functions.
//
// 3. `coerce_value(ctx, val, from, to)` — Convert between arbitrary types.
//    Handles Any↔Concrete, Int↔Float. Use for assignment/return coercion.
//
// 4. `unwrap_any_ptr(ctx, val, ty)` — Extract raw pointer from Any TokValue.
//    Use when a builtin expects a concrete pointer (e.g., array/map) but the
//    HIR type is Any. No-op if ty is already concrete.
//
// 5. `alloc_tokvalue_on_stack(ctx, tag, data)` — Store (tag, data) as a
//    stack-allocated TokValue and return the pointer. Use when wrapping
//    concrete values into Any representation.
//
// Convention: Any-typed values are always PTR to a 16-byte stack slot
// (tag @ offset 0, data @ offset 8).

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value};
use cranelift_frontend::FunctionBuilder;
use tok_types::Type;
use tok_hir::hir::*;
use super::{FuncCtx, PTR, compile_expr};

/// TAG constants matching the runtime.
pub(crate) const TAG_NIL: i64 = 0;
pub(crate) const TAG_INT: i64 = 1;
pub(crate) const TAG_FLOAT: i64 = 2;
pub(crate) const TAG_BOOL: i64 = 3;
pub(crate) const TAG_STRING: i64 = 4;
pub(crate) const TAG_ARRAY: i64 = 5;
pub(crate) const TAG_MAP: i64 = 6;
pub(crate) const TAG_TUPLE: i64 = 7;
pub(crate) const TAG_FUNC: i64 = 8;
pub(crate) const TAG_CHANNEL: i64 = 9;
pub(crate) const TAG_HANDLE: i64 = 10;

/// Allocate a TokValue on the stack and store tag+data, returning a pointer.
/// This is used to wrap concrete values into the Any representation.
pub(crate) fn alloc_tokvalue_on_stack(ctx: &mut FuncCtx, tag: Value, data: Value) -> Value {
    let ss = ctx.builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        16,
        3, // 8-byte alignment
    ));
    let addr = ctx.builder.ins().stack_addr(PTR, ss, 0);
    ctx.builder.ins().store(MemFlags::trusted(), tag, addr, 0);
    ctx.builder.ins().store(MemFlags::trusted(), data, addr, 8);
    addr
}

/// Coerce a value from one type to another.
/// E.g., Int → Any: wrap in stack-allocated TokValue. Any → Int: load and extract.
pub(crate) fn coerce_value(ctx: &mut FuncCtx, val: Value, from: &Type, to: &Type) -> Value {
    // Same type or both non-Any → no coercion needed
    if std::mem::discriminant(from) == std::mem::discriminant(to) {
        return val;
    }
    // Concrete → Any: wrap in stack TokValue
    if matches!(to, Type::Any) && !matches!(from, Type::Any) {
        let (tag, data) = to_tokvalue(ctx, val, from);
        return from_tokvalue(ctx, tag, data, &Type::Any);
    }
    // Any → Concrete: unwrap from stack TokValue
    if matches!(from, Type::Any) && !matches!(to, Type::Any) {
        let (tag, data) = to_tokvalue(ctx, val, from);
        return from_tokvalue(ctx, tag, data, to);
    }
    // Int → Float
    if matches!(from, Type::Int) && matches!(to, Type::Float) {
        return ctx.builder.ins().fcvt_from_sint(types::F64, val);
    }
    // Float → Int
    if matches!(from, Type::Float) && matches!(to, Type::Int) {
        return ctx.builder.ins().fcvt_to_sint_sat(types::I64, val);
    }
    val
}

/// If the expr type is Any, extract the raw pointer (data field at offset 8) from the
/// TokValue. Otherwise return the value as-is (it's already a raw pointer).
/// Used for builtins that expect a concrete pointer (Array, Map, String, etc.)
/// but may receive an Any-typed TokValue pointer.
pub(crate) fn unwrap_any_ptr(ctx: &mut FuncCtx, val: Value, ty: &Type) -> Value {
    if matches!(ty, Type::Any) {
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), val, 8)
    } else {
        val
    }
}

/// Extract the raw i64 data field from a stack-allocated TokValue pointer.
///
/// TokValue layout: tag (i64) at offset 0, data (i64) at offset 8.
/// Use when the caller needs the untagged payload (e.g., an int exit code
/// or channel capacity) from an Any-typed value.
pub(crate) fn from_tokvalue_raw_data(ctx: &mut FuncCtx, tokvalue_ptr: Value) -> Value {
    ctx.builder
        .ins()
        .load(types::I64, MemFlags::trusted(), tokvalue_ptr, 8)
}

/// Compile an expression and extract its raw pointer, unwrapping from Any if needed.
///
/// Shorthand for `compile_expr(ctx, expr).unwrap() + unwrap_any_ptr(ctx, val, &expr.ty)`.
/// Use when the caller needs a concrete pointer (e.g., array, map, string) from an
/// expression that may be typed as `Any`.
pub(crate) fn compile_expr_as_ptr(ctx: &mut FuncCtx, expr: &HirExpr) -> Value {
    let val = compile_expr(ctx, expr).expect("codegen: expression produced no value");
    unwrap_any_ptr(ctx, val, &expr.ty)
}

// ─── TokValue packing/unpacking ───────────────────────────────────────

/// Pack a typed value into (tag: I64, data: I64) for runtime calls that take TokValue.
pub(crate) fn to_tokvalue(ctx: &mut FuncCtx, val: Value, ty: &Type) -> (Value, Value) {
    // Normalize: if actual Cranelift type is i8 but Tok type is not Bool,
    // the value is a boolean that was mistyped (e.g., `|` is logical OR returning i8,
    // but the result was assigned to a variable typed as Int). Widen to i64.
    let val = {
        let actual = ctx.builder.func.dfg.value_type(val);
        if actual == types::I8 && !matches!(ty, Type::Bool) {
            ctx.builder.ins().uextend(types::I64, val)
        } else {
            val
        }
    };
    let (tag, data) = match ty {
        Type::Int => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_INT);
            (tag, val) // i64 fits in data word
        }
        Type::Float => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_FLOAT);
            let bits = ctx.builder.ins().bitcast(types::I64, MemFlags::new(), val);
            (tag, bits)
        }
        Type::Bool => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_BOOL);
            let ext = ctx.builder.ins().uextend(types::I64, val);
            (tag, ext)
        }
        Type::Str => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_STRING);
            (tag, val)
        }
        Type::Array(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_ARRAY);
            (tag, val)
        }
        Type::Map(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_MAP);
            (tag, val)
        }
        Type::Tuple(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_TUPLE);
            (tag, val)
        }
        Type::Func(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_FUNC);
            (tag, val)
        }
        Type::Channel(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_CHANNEL);
            (tag, val)
        }
        Type::Handle(_) => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_HANDLE);
            (tag, val)
        }
        Type::Nil | Type::Never => {
            let tag = ctx.builder.ins().iconst(PTR, TAG_NIL);
            let data = ctx.builder.ins().iconst(types::I64, 0);
            (tag, data)
        }
        Type::Any | Type::Optional(_) | Type::Result(_) | Type::Range => {
            // Check actual Cranelift type — HIR may say Any but the value could be
            // a native i8 (bool from short-circuit and/or) or f64 (float).
            let actual_ty = ctx.builder.func.dfg.value_type(val);
            if actual_ty == types::I8 {
                let tag = ctx.builder.ins().iconst(PTR, TAG_BOOL);
                let ext = ctx.builder.ins().uextend(types::I64, val);
                return (tag, ext);
            }
            if actual_ty == types::F64 {
                let tag = ctx.builder.ins().iconst(PTR, TAG_FLOAT);
                let bits = ctx.builder.ins().bitcast(types::I64, MemFlags::new(), val);
                return (tag, bits);
            }
            // `Any` values are stored as stack-allocated TokValues (ptr to 16-byte struct).
            // Load tag (offset 0) and data (offset 8) from the stack slot.
            let tag = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), val, 0);
            let data = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), val, 8);
            return (tag, data);
        }
    };
    (tag, data)
}

/// Unpack a (tag: I64, data: I64) TokValue into the expected typed value.
pub(crate) fn from_tokvalue(ctx: &mut FuncCtx, tag: Value, data: Value, ty: &Type) -> Value {
    match ty {
        Type::Int => data,
        Type::Float => ctx.builder.ins().bitcast(types::F64, MemFlags::new(), data),
        Type::Bool => ctx.builder.ins().ireduce(types::I8, data),
        Type::Str
        | Type::Array(_)
        | Type::Map(_)
        | Type::Tuple(_)
        | Type::Func(_)
        | Type::Channel(_)
        | Type::Handle(_) => data, // pointer
        Type::Any | Type::Optional(_) | Type::Result(_) | Type::Range => {
            alloc_tokvalue_on_stack(ctx, tag, data)
        }
        _ => data,
    }
}

/// Convert a value to a boolean (i8) for branching.
pub(crate) fn to_bool(ctx: &mut FuncCtx, val: Value, ty: &Type) -> Value {
    match ty {
        Type::Bool => val,
        Type::Int => {
            let zero = ctx.builder.ins().iconst(types::I64, 0);
            ctx.builder
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, val, zero)
        }
        Type::Float => {
            let zero = ctx.builder.ins().f64const(0.0);
            ctx.builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                val,
                zero,
            )
        }
        Type::Str | Type::Array(_) | Type::Map(_) => {
            // Match interpreter: empty string/array/map is falsy.
            // Delegate to runtime truthiness check via tag+data.
            let (tag, data) = to_tokvalue(ctx, val, ty);
            let func_ref = ctx.get_runtime_func_ref("tok_value_truthiness");
            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
            ctx.builder.inst_results(call)[0]
        }
        Type::Tuple(_) => {
            // Tuples are always truthy (non-null pointer)
            let zero = ctx.builder.ins().iconst(PTR, 0);
            ctx.builder
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, val, zero)
        }
        _ => {
            // If the actual Cranelift type is already i8 (bool), return as-is
            let actual_ty = ctx.builder.func.dfg.value_type(val);
            if actual_ty == types::I8 {
                return val;
            }
            // For Any types, use runtime truthiness
            let (tag, data) = to_tokvalue(ctx, val, ty);
            let func_ref = ctx.get_runtime_func_ref("tok_value_truthiness");
            let call = ctx.builder.ins().call(func_ref, &[tag, data]);
            ctx.builder.inst_results(call)[0]
        }
    }
}

/// Create a zero/default value of the given Cranelift type.
/// Coerce an if-branch value to match the merge block's expected type.
/// Handles type mismatches between branch values and the merge block parameter.
pub(crate) fn coerce_if_branch(
    ctx: &mut FuncCtx,
    val: Value,
    branch_expr_ty: Option<&Type>,
    result_ty: &Type,
    result_cl_type: types::Type,
) -> Value {
    if let Some(branch_ty) = branch_expr_ty {
        // If merge block is Any: wrap concrete branch values as TokValue
        if matches!(result_ty, Type::Any) && !matches!(branch_ty, Type::Any) {
            let (tag, data) = to_tokvalue(ctx, val, branch_ty);
            return alloc_tokvalue_on_stack(ctx, tag, data);
        }
        // Both are Any: no coercion needed
        if matches!(result_ty, Type::Any) && matches!(branch_ty, Type::Any) {
            return val;
        }
    }
    coerce_cl_value(ctx, val, result_cl_type)
}

/// Coerce a Cranelift value to the target type by inserting uextend/bitcast as needed.
pub(crate) fn coerce_cl_value(ctx: &mut FuncCtx, val: Value, target: types::Type) -> Value {
    let actual = ctx.builder.func.dfg.value_type(val);
    if actual == target {
        return val;
    }
    // i8 → i64: uextend (Bool → Int/Any)
    if actual == types::I8 && target == types::I64 {
        return ctx.builder.ins().uextend(types::I64, val);
    }
    // f64 → i64: bitcast (Float → Any)
    if actual == types::F64 && target == types::I64 {
        return ctx.builder.ins().bitcast(types::I64, MemFlags::new(), val);
    }
    // i64 → f64: bitcast (Any → Float)
    if actual == types::I64 && target == types::F64 {
        return ctx.builder.ins().bitcast(types::F64, MemFlags::new(), val);
    }
    // i64 → i8: ireduce (Int → Bool)
    if actual == types::I64 && target == types::I8 {
        return ctx.builder.ins().ireduce(types::I8, val);
    }
    val
}

pub(crate) fn zero_value(builder: &mut FunctionBuilder, ty: types::Type) -> Value {
    if ty == types::F64 {
        builder.ins().f64const(0.0)
    } else if ty == types::F32 {
        builder.ins().f32const(0.0)
    } else {
        builder.ins().iconst(ty, 0)
    }
}
