//! Shared helpers for stdlib trampoline functions.
//!
//! These extract typed values from the (tag, data) pairs passed by codegen.
//! Each trampoline receives arguments as `(tag: i64, data: i64)` — these
//! helpers decode the tag and reinterpret the data field.

use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{safe_f64_to_i64, TokValue, TAG_FLOAT, TAG_INT, TAG_STRING};

/// Insert a function entry into a stdlib module map.
///
/// Allocates a closure wrapping `fn_ptr` with the given `arity` and inserts
/// it into the map under `name`. Used by all stdlib module constructors.
pub fn insert_func(m: *mut TokMap, name: &str, fn_ptr: *const u8, arity: u32) {
    let closure = TokClosure::alloc(fn_ptr, std::ptr::null_mut(), arity, 0);
    let val = TokValue::from_func(closure);
    unsafe {
        (*m).data.insert(name.to_string(), val);
    }
}

/// Extract a `&str` from a (tag, data) pair.
///
/// # Safety
///
/// The caller must ensure that if `tag == TAG_STRING`, the `data` field
/// contains a valid, non-null `*mut TokString` whose pointee will not be
/// freed for the duration of the returned borrow. In practice this is
/// guaranteed because:
///
/// - The caller holds the (tag, data) values on the stack
/// - No concurrent code can free the TokString while this function's
///   caller is executing (refcount >= 1 for the caller's copy)
/// - The returned `&str` is used and dropped within the same `extern "C"`
///   trampoline function
#[inline]
pub unsafe fn arg_to_str(tag: i64, data: i64) -> &'static str {
    if tag as u8 == TAG_STRING {
        let ptr = data as *mut TokString;
        if !ptr.is_null() {
            // Safety: ptr is valid for the duration of the caller's stack frame.
            // We return 'static to avoid needing a borrow source for the lifetime,
            // but the caller must not store this reference beyond the current
            // trampoline invocation.
            return &(*ptr).data;
        }
    }
    ""
}

/// Extract an `i64` from a (tag, data) pair. Coerces float→int.
#[inline]
pub fn arg_to_i64(tag: i64, data: i64) -> i64 {
    match tag as u8 {
        TAG_INT => data,
        TAG_FLOAT => safe_f64_to_i64(f64::from_bits(data as u64)),
        _ => 0,
    }
}

/// Extract an `f64` from a (tag, data) pair. Coerces int→float.
#[inline]
pub fn arg_to_f64(tag: i64, data: i64) -> f64 {
    match tag as u8 {
        TAG_FLOAT => f64::from_bits(data as u64),
        TAG_INT => data as f64,
        _ => 0.0,
    }
}

/// Generate a 1-arg `f64 → f64` trampoline function.
///
/// Produces a `#[no_mangle] pub extern "C" fn $name(_env, tag, data) -> TokValue`
/// that calls `arg_to_f64(tag, data).$method()` and wraps the result.
///
/// Usage: `math_f64_unary!(tok_math_sqrt_t, sqrt);`
#[macro_export]
macro_rules! math_f64_unary {
    ($name:ident, $method:ident) => {
        #[no_mangle]
        pub extern "C" fn $name(_env: *mut u8, tag: i64, data: i64) -> $crate::value::TokValue {
            $crate::value::TokValue::from_float(
                $crate::stdlib_helpers::arg_to_f64(tag, data).$method(),
            )
        }
    };
}

/// Generate a 1-arg `f64 → i64` trampoline (floor/ceil/round).
///
/// Usage: `math_f64_to_int!(tok_math_floor_t, floor);`
#[macro_export]
macro_rules! math_f64_to_int {
    ($name:ident, $method:ident) => {
        #[no_mangle]
        pub extern "C" fn $name(_env: *mut u8, tag: i64, data: i64) -> $crate::value::TokValue {
            $crate::value::TokValue::from_int($crate::value::safe_f64_to_i64(
                $crate::stdlib_helpers::arg_to_f64(tag, data).$method(),
            ))
        }
    };
}

/// Generate a 2-arg `(f64, f64) → f64` trampoline.
///
/// Usage: `math_f64_binary!(tok_math_pow_t, powf);`
#[macro_export]
macro_rules! math_f64_binary {
    ($name:ident, $method:ident) => {
        #[no_mangle]
        pub extern "C" fn $name(
            _env: *mut u8,
            tag1: i64,
            data1: i64,
            tag2: i64,
            data2: i64,
        ) -> $crate::value::TokValue {
            let a = $crate::stdlib_helpers::arg_to_f64(tag1, data1);
            let b = $crate::stdlib_helpers::arg_to_f64(tag2, data2);
            $crate::value::TokValue::from_float(a.$method(b))
        }
    };
}

/// Generate a 1-arg `str → str` trampoline.
///
/// Calls `arg_to_str(tag, data).$method()` and wraps as TokString.
/// The method must return something that implements `ToString`.
///
/// Usage: `str_unary!(tok_str_upper_t, to_uppercase);`
#[macro_export]
macro_rules! str_unary {
    ($name:ident, $method:ident) => {
        #[no_mangle]
        pub extern "C" fn $name(_env: *mut u8, tag: i64, data: i64) -> $crate::value::TokValue {
            unsafe {
                let s = $crate::stdlib_helpers::arg_to_str(tag, data);
                $crate::value::TokValue::from_string($crate::string::TokString::alloc(
                    s.$method().to_string(),
                ))
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string::TokString;

    #[test]
    fn test_arg_to_str_valid() {
        let s = TokString::alloc("hello".to_string());
        unsafe {
            let result = arg_to_str(TAG_STRING as i64, s as i64);
            assert_eq!(result, "hello");
            drop(Box::from_raw(s));
        }
    }

    #[test]
    fn test_arg_to_str_null() {
        unsafe {
            let result = arg_to_str(TAG_STRING as i64, 0);
            assert_eq!(result, "");
        }
    }

    #[test]
    fn test_arg_to_str_wrong_tag() {
        unsafe {
            let result = arg_to_str(TAG_INT as i64, 42);
            assert_eq!(result, "");
        }
    }

    #[test]
    fn test_arg_to_i64_int() {
        assert_eq!(arg_to_i64(TAG_INT as i64, 42), 42);
    }

    #[test]
    fn test_arg_to_i64_float() {
        let f: f64 = 3.7;
        assert_eq!(arg_to_i64(TAG_FLOAT as i64, f.to_bits() as i64), 3);
    }

    #[test]
    fn test_arg_to_i64_wrong_tag() {
        assert_eq!(arg_to_i64(TAG_STRING as i64, 999), 0);
    }

    #[test]
    fn test_arg_to_f64_float() {
        let f: f64 = 3.14;
        let result = arg_to_f64(TAG_FLOAT as i64, f.to_bits() as i64);
        assert!((result - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn test_arg_to_f64_int() {
        let result = arg_to_f64(TAG_INT as i64, 5);
        assert!((result - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_arg_to_f64_wrong_tag() {
        assert!((arg_to_f64(TAG_STRING as i64, 0)).abs() < f64::EPSILON);
    }
}
