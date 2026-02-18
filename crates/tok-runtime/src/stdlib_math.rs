//! Standard library: `@"math"` module.
//!
//! Provides mathematical functions and constants as a TokMap of closures.

use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{TokValue, TAG_FLOAT, TAG_INT};

use std::f64::consts;

use crate::stdlib_helpers::arg_to_f64;

// ═══════════════════════════════════════════════════════════════
// Trampolines — closure ABI: (env, tag, data, ...) -> TokValue
// ═══════════════════════════════════════════════════════════════

// --- 1-arg Float -> Float ---

#[no_mangle]
pub extern "C" fn tok_math_sqrt_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).sqrt())
}

#[no_mangle]
pub extern "C" fn tok_math_sin_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).sin())
}

#[no_mangle]
pub extern "C" fn tok_math_cos_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).cos())
}

#[no_mangle]
pub extern "C" fn tok_math_tan_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).tan())
}

#[no_mangle]
pub extern "C" fn tok_math_asin_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).asin())
}

#[no_mangle]
pub extern "C" fn tok_math_acos_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).acos())
}

#[no_mangle]
pub extern "C" fn tok_math_atan_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).atan())
}

#[no_mangle]
pub extern "C" fn tok_math_log_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).ln())
}

#[no_mangle]
pub extern "C" fn tok_math_log2_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).log2())
}

#[no_mangle]
pub extern "C" fn tok_math_log10_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).log10())
}

#[no_mangle]
pub extern "C" fn tok_math_exp_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_float(arg_to_f64(tag, data).exp())
}

// --- 1-arg Float -> Int ---

#[no_mangle]
pub extern "C" fn tok_math_floor_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_int(arg_to_f64(tag, data).floor() as i64)
}

#[no_mangle]
pub extern "C" fn tok_math_ceil_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_int(arg_to_f64(tag, data).ceil() as i64)
}

#[no_mangle]
pub extern "C" fn tok_math_round_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    TokValue::from_int(arg_to_f64(tag, data).round() as i64)
}

// --- 1-arg abs (preserves type) ---

#[no_mangle]
pub extern "C" fn tok_math_abs_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    match tag as u8 {
        TAG_INT => TokValue::from_int(data.abs()),
        TAG_FLOAT => TokValue::from_float(f64::from_bits(data as u64).abs()),
        _ => TokValue::from_int(0),
    }
}

// --- 2-arg Float, Float -> Float ---

#[no_mangle]
pub extern "C" fn tok_math_pow_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    let base = arg_to_f64(tag1, data1);
    let exp = arg_to_f64(tag2, data2);
    TokValue::from_float(base.powf(exp))
}

#[no_mangle]
pub extern "C" fn tok_math_atan2_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    let y = arg_to_f64(tag1, data1);
    let x = arg_to_f64(tag2, data2);
    TokValue::from_float(y.atan2(x))
}

// --- 2-arg min/max (preserves type) ---

#[no_mangle]
pub extern "C" fn tok_math_min_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    let a = arg_to_f64(tag1, data1);
    let b = arg_to_f64(tag2, data2);
    if a <= b {
        TokValue::from_tag_data(tag1, data1)
    } else {
        TokValue::from_tag_data(tag2, data2)
    }
}

#[no_mangle]
pub extern "C" fn tok_math_max_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    let a = arg_to_f64(tag1, data1);
    let b = arg_to_f64(tag2, data2);
    if a >= b {
        TokValue::from_tag_data(tag1, data1)
    } else {
        TokValue::from_tag_data(tag2, data2)
    }
}

// --- 0-arg random ---

#[no_mangle]
pub extern "C" fn tok_math_random_t(_env: *mut u8) -> TokValue {
    // Simple xorshift-based random, seeded from time
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0);
    let mut s = STATE.load(Ordering::Relaxed);
    if s == 0 {
        s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        if s == 0 {
            s = 1;
        }
    }
    // xorshift64
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    STATE.store(s, Ordering::Relaxed);
    // Convert to [0, 1)
    let f = (s >> 11) as f64 / (1u64 << 53) as f64;
    TokValue::from_float(f)
}

// ═══════════════════════════════════════════════════════════════
// Module constructor
// ═══════════════════════════════════════════════════════════════

fn insert_func(m: *mut TokMap, name: &str, fn_ptr: *const u8, arity: u32) {
    let key = TokString::alloc(name.to_string());
    let closure = TokClosure::alloc(fn_ptr, std::ptr::null_mut(), arity);
    let val = TokValue::from_func(closure);
    unsafe {
        // Direct insert — skip tok_map_set's rc_inc because val starts at rc=1
        (*m).data.insert(name.to_string(), val);
    }
    // Free the key we used just for the name
    unsafe {
        drop(Box::from_raw(key));
    }
}

fn insert_float(m: *mut TokMap, name: &str, val: f64) {
    unsafe {
        (*m).data
            .insert(name.to_string(), TokValue::from_float(val));
    }
}

#[no_mangle]
pub extern "C" fn tok_stdlib_math() -> *mut TokMap {
    let m = TokMap::alloc();

    // 1-arg functions
    insert_func(m, "sqrt", tok_math_sqrt_t as *const u8, 1);
    insert_func(m, "sin", tok_math_sin_t as *const u8, 1);
    insert_func(m, "cos", tok_math_cos_t as *const u8, 1);
    insert_func(m, "tan", tok_math_tan_t as *const u8, 1);
    insert_func(m, "asin", tok_math_asin_t as *const u8, 1);
    insert_func(m, "acos", tok_math_acos_t as *const u8, 1);
    insert_func(m, "atan", tok_math_atan_t as *const u8, 1);
    insert_func(m, "log", tok_math_log_t as *const u8, 1);
    insert_func(m, "log2", tok_math_log2_t as *const u8, 1);
    insert_func(m, "log10", tok_math_log10_t as *const u8, 1);
    insert_func(m, "exp", tok_math_exp_t as *const u8, 1);
    insert_func(m, "floor", tok_math_floor_t as *const u8, 1);
    insert_func(m, "ceil", tok_math_ceil_t as *const u8, 1);
    insert_func(m, "round", tok_math_round_t as *const u8, 1);
    insert_func(m, "abs", tok_math_abs_t as *const u8, 1);

    // 2-arg functions
    insert_func(m, "pow", tok_math_pow_t as *const u8, 2);
    insert_func(m, "min", tok_math_min_t as *const u8, 2);
    insert_func(m, "max", tok_math_max_t as *const u8, 2);
    insert_func(m, "atan2", tok_math_atan2_t as *const u8, 2);

    // 0-arg functions
    insert_func(m, "random", tok_math_random_t as *const u8, 0);

    // Constants (stored as float values directly, not closures)
    insert_float(m, "pi", consts::PI);
    insert_float(m, "e", consts::E);
    insert_float(m, "inf", f64::INFINITY);
    insert_float(m, "nan", f64::NAN);

    m
}
