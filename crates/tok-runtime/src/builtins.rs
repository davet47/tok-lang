//! Built-in `extern "C"` functions for the Tok runtime.
//!
//! These are called directly by Cranelift-generated machine code.

use crate::array::tok_array_slice;
use crate::string::{tok_string_repeat, tok_string_slice, TokString};
use crate::value::{
    format_float, safe_f64_to_i64, TokValue, TAG_ARRAY, TAG_BOOL, TAG_FLOAT, TAG_INT, TAG_NIL,
    TAG_STRING,
};

// ═══════════════════════════════════════════════════════════════
// Reference counting (generic)
// ═══════════════════════════════════════════════════════════════

/// Increment the refcount of any Tok heap object.
/// All Tok heap objects start with an `AtomicU32` refcount field.
#[no_mangle]
pub extern "C" fn tok_rc_inc(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let rc = &*(ptr as *const std::sync::atomic::AtomicU32);
        rc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Decrement the refcount of any Tok heap object.
///
/// **Important**: This only handles the refcount decrement. The codegen
/// must pair this with type-specific cleanup (calling `Box::from_raw`)
/// when the count reaches zero. For generic use, prefer the typed
/// `TokValue::rc_dec()`.
///
/// Returns 1 if the object should be freed (refcount reached 0).
#[no_mangle]
pub extern "C" fn tok_rc_dec(ptr: *mut u8) -> i8 {
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        let rc = &*(ptr as *const std::sync::atomic::AtomicU32);
        if rc.fetch_sub(1, std::sync::atomic::Ordering::Release) == 1 {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            1
        } else {
            0
        }
    }
}

/// Full TokValue-level rc_dec: reconstructs a TokValue from (tag, data)
/// and calls its rc_dec() method, which handles recursive child cleanup
/// and deallocation when refcount reaches zero.
/// Called by codegen on variable reassignment and function exit.
///
/// The (tag, data) pair matches the codegen's `to_tokvalue` representation:
/// tag is the type discriminant (0-10) stored as i64, data is the payload.
#[no_mangle]
pub extern "C" fn tok_value_rc_dec(tag: i64, data: i64) {
    // Reconstruct TokValue from its raw (tag, data) codegen representation.
    let v = TokValue::from_tag_data(tag, data);
    v.rc_dec();
}

/// Fast rc_dec for strings: skips TokValue reconstruction overhead.
/// Called by codegen when the variable type is known to be Str.
#[no_mangle]
pub extern "C" fn tok_string_free(ptr: *mut TokString) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        if (*ptr).rc_dec() {
            drop(Box::from_raw(ptr));
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Print builtins
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_print(val: TokValue) {
    print!("{}", val);
}

#[no_mangle]
pub extern "C" fn tok_println(val: TokValue) {
    println!("{}", val);
}

#[no_mangle]
pub extern "C" fn tok_print_int(val: i64) {
    print!("{}", val);
}

#[no_mangle]
pub extern "C" fn tok_println_int(val: i64) {
    println!("{}", val);
}

#[no_mangle]
pub extern "C" fn tok_print_float(val: f64) {
    print!("{}", format_float(val));
}

#[no_mangle]
pub extern "C" fn tok_println_float(val: f64) {
    println!("{}", format_float(val));
}

#[no_mangle]
pub extern "C" fn tok_print_string(val: *mut TokString) {
    if val.is_null() {
        print!("N");
    } else {
        unsafe {
            print!("{}", (*val).data);
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_println_string(val: *mut TokString) {
    if val.is_null() {
        println!("N");
    } else {
        unsafe {
            println!("{}", (*val).data);
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_print_bool(val: i8) {
    print!("{}", if val != 0 { "T" } else { "F" });
}

#[no_mangle]
pub extern "C" fn tok_println_bool(val: i8) {
    println!("{}", if val != 0 { "T" } else { "F" });
}

// ═══════════════════════════════════════════════════════════════
// Type inspection
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_type_of(val: TokValue) -> *mut TokString {
    TokString::alloc(val.type_name().to_string())
}

// ═══════════════════════════════════════════════════════════════
// Utility
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_clock() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[no_mangle]
pub extern "C" fn tok_exit(code: i64) {
    std::process::exit(code as i32);
}

// ═══════════════════════════════════════════════════════════════
// Conversion to string
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_int_to_string(val: i64) -> *mut TokString {
    TokString::alloc(format!("{}", val))
}

#[no_mangle]
pub extern "C" fn tok_float_to_string(val: f64) -> *mut TokString {
    TokString::alloc(format_float(val))
}

#[no_mangle]
pub extern "C" fn tok_bool_to_string(val: i8) -> *mut TokString {
    TokString::alloc(if val != 0 {
        "T".to_string()
    } else {
        "F".to_string()
    })
}

#[no_mangle]
pub extern "C" fn tok_value_to_string(val: TokValue) -> *mut TokString {
    TokString::alloc(format!("{}", val))
}

// ═══════════════════════════════════════════════════════════════
// Numeric builtins
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_abs_int(val: i64) -> i64 {
    val.saturating_abs()
}

#[no_mangle]
pub extern "C" fn tok_abs_float(val: f64) -> f64 {
    val.abs()
}

/// abs() for Any-typed values: dispatches by tag, returns TokValue.
#[no_mangle]
pub extern "C" fn tok_value_abs(val: TokValue) -> TokValue {
    unsafe {
        match val.tag {
            TAG_INT => TokValue::from_int(val.data.int_val.saturating_abs()),
            TAG_FLOAT => TokValue::from_float(val.data.float_val.abs()),
            _ => val,
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_floor(val: f64) -> i64 {
    safe_f64_to_i64(val.floor())
}

/// floor() for Any-typed values: dispatches by tag, returns TokValue.
#[no_mangle]
pub extern "C" fn tok_value_floor(val: TokValue) -> TokValue {
    unsafe {
        match val.tag {
            TAG_FLOAT => TokValue::from_int(safe_f64_to_i64(val.data.float_val.floor())),
            TAG_INT => val,
            _ => val,
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_ceil(val: f64) -> i64 {
    safe_f64_to_i64(val.ceil())
}

/// ceil() for Any-typed values: dispatches by tag, returns TokValue.
#[no_mangle]
pub extern "C" fn tok_value_ceil(val: TokValue) -> TokValue {
    unsafe {
        match val.tag {
            TAG_FLOAT => TokValue::from_int(safe_f64_to_i64(val.data.float_val.ceil())),
            TAG_INT => val,
            _ => val,
        }
    }
}

// slice() for Any-typed values: dispatches by tag (array or string), returns TokValue.
#[no_mangle]
pub extern "C" fn tok_value_slice(val: TokValue, start: i64, end: i64) -> TokValue {
    unsafe {
        match val.tag {
            TAG_ARRAY => {
                let result = tok_array_slice(val.data.array_ptr, start, end);
                TokValue::from_array(result)
            }
            TAG_STRING => {
                let result = tok_string_slice(val.data.string_ptr, start, end);
                TokValue::from_string(result)
            }
            _ => TokValue::nil(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// TokValue → concrete type extraction
// ═══════════════════════════════════════════════════════════════

/// Extract an i64 from a TokValue. Coerces Float→Int, Bool→Int. Returns 0 for other types.
#[no_mangle]
pub extern "C" fn tok_value_to_int(val: TokValue) -> i64 {
    unsafe {
        match val.tag {
            TAG_INT => val.data.int_val,
            TAG_FLOAT => val.data.float_val as i64,
            TAG_BOOL => val.data.bool_val as i64,
            _ => 0,
        }
    }
}

/// Extract an f64 from a TokValue. Coerces Int→Float, Bool→Float. Returns 0.0 for other types.
#[no_mangle]
pub extern "C" fn tok_value_to_float(val: TokValue) -> f64 {
    unsafe {
        match val.tag {
            TAG_FLOAT => val.data.float_val,
            TAG_INT => val.data.int_val as f64,
            TAG_BOOL => val.data.bool_val as f64,
            _ => 0.0,
        }
    }
}

/// Extract an i8 (bool) from a TokValue. Uses truthiness semantics.
#[no_mangle]
pub extern "C" fn tok_value_to_bool(val: TokValue) -> i8 {
    unsafe {
        match val.tag {
            TAG_BOOL => val.data.bool_val,
            TAG_INT => {
                if val.data.int_val != 0 {
                    1
                } else {
                    0
                }
            }
            TAG_FLOAT => {
                if val.data.float_val != 0.0 {
                    1
                } else {
                    0
                }
            }
            TAG_NIL => 0,
            TAG_STRING => {
                if !val.data.string_ptr.is_null() && !(*val.data.string_ptr).data.is_empty() {
                    1
                } else {
                    0
                }
            }
            _ => 1, // Arrays, maps, tuples, etc. are truthy
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_rand() -> f64 {
    // Simple xorshift-based PRNG seeded from system time.
    // No external dependency needed.
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            // Make sure seed is never 0 for xorshift
            if seed == 0 { 1 } else { seed }
        });
    }

    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        // Convert to [0.0, 1.0)
        (x >> 11) as f64 / (1u64 << 53) as f64
    })
}

#[no_mangle]
pub extern "C" fn tok_to_int(val: TokValue) -> i64 {
    unsafe {
        match val.tag {
            TAG_INT => val.data.int_val,
            TAG_FLOAT => val.data.float_val as i64,
            TAG_BOOL => val.data.bool_val as i64,
            TAG_STRING => {
                if val.data.string_ptr.is_null() {
                    0
                } else {
                    (*val.data.string_ptr)
                        .data
                        .trim()
                        .parse::<i64>()
                        .unwrap_or(0)
                }
            }
            _ => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_to_float(val: TokValue) -> f64 {
    unsafe {
        match val.tag {
            TAG_INT => val.data.int_val as f64,
            TAG_FLOAT => val.data.float_val,
            TAG_BOOL => val.data.bool_val as f64,
            TAG_STRING => {
                if val.data.string_ptr.is_null() {
                    0.0
                } else {
                    (*val.data.string_ptr)
                        .data
                        .trim()
                        .parse::<f64>()
                        .unwrap_or(0.0)
                }
            }
            _ => 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// TokValue dynamic dispatch operations (for `a` type)
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_value_add(a: TokValue, b: TokValue) -> TokValue {
    unsafe {
        match (a.tag, b.tag) {
            (TAG_INT, TAG_INT) => TokValue::from_int(a.data.int_val.wrapping_add(b.data.int_val)),
            (TAG_FLOAT, TAG_FLOAT) => TokValue::from_float(a.data.float_val + b.data.float_val),
            (TAG_INT, TAG_FLOAT) => TokValue::from_float(a.data.int_val as f64 + b.data.float_val),
            (TAG_FLOAT, TAG_INT) => TokValue::from_float(a.data.float_val + b.data.int_val as f64),
            (TAG_STRING, TAG_STRING) => {
                let ap = a.data.string_ptr;
                let bp = b.data.string_ptr;
                // COW: if `a` string has refcount 1, mutate in-place.
                // Acquire synchronizes with Release in rc_dec.
                if !ap.is_null() && (*ap).rc.load(std::sync::atomic::Ordering::Acquire) == 1 {
                    let sb = if bp.is_null() { "" } else { &(*bp).data };
                    (*ap).data.push_str(sb);
                    a // return the same TokValue (same pointer, mutated content)
                } else {
                    let sa = if ap.is_null() { "" } else { &(*ap).data };
                    let sb = if bp.is_null() { "" } else { &(*bp).data };
                    let mut result = String::with_capacity(sa.len() + sb.len());
                    result.push_str(sa);
                    result.push_str(sb);
                    TokValue::from_string(TokString::alloc(result))
                }
            }
            // String + other → concat with string representation
            (TAG_STRING, _) => {
                let sa = if a.data.string_ptr.is_null() {
                    String::new()
                } else {
                    (*a.data.string_ptr).data.clone()
                };
                TokValue::from_string(TokString::alloc(format!("{}{}", sa, b)))
            }
            (_, TAG_STRING) => {
                let sb = if b.data.string_ptr.is_null() {
                    String::new()
                } else {
                    (*b.data.string_ptr).data.clone()
                };
                TokValue::from_string(TokString::alloc(format!("{}{}", a, sb)))
            }
            _ => TokValue::nil(),
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_value_sub(a: TokValue, b: TokValue) -> TokValue {
    unsafe {
        match (a.tag, b.tag) {
            (TAG_INT, TAG_INT) => TokValue::from_int(a.data.int_val.wrapping_sub(b.data.int_val)),
            (TAG_FLOAT, TAG_FLOAT) => TokValue::from_float(a.data.float_val - b.data.float_val),
            (TAG_INT, TAG_FLOAT) => TokValue::from_float(a.data.int_val as f64 - b.data.float_val),
            (TAG_FLOAT, TAG_INT) => TokValue::from_float(a.data.float_val - b.data.int_val as f64),
            _ => TokValue::nil(),
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_value_mul(a: TokValue, b: TokValue) -> TokValue {
    unsafe {
        match (a.tag, b.tag) {
            (TAG_INT, TAG_INT) => TokValue::from_int(a.data.int_val.wrapping_mul(b.data.int_val)),
            (TAG_FLOAT, TAG_FLOAT) => TokValue::from_float(a.data.float_val * b.data.float_val),
            (TAG_INT, TAG_FLOAT) => TokValue::from_float(a.data.int_val as f64 * b.data.float_val),
            (TAG_FLOAT, TAG_INT) => TokValue::from_float(a.data.float_val * b.data.int_val as f64),
            (TAG_STRING, TAG_INT) => {
                let result = tok_string_repeat(a.data.string_ptr, b.data.int_val);
                TokValue::from_string(result)
            }
            (TAG_INT, TAG_STRING) => {
                let result = tok_string_repeat(b.data.string_ptr, a.data.int_val);
                TokValue::from_string(result)
            }
            _ => TokValue::nil(),
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_value_div(a: TokValue, b: TokValue) -> TokValue {
    unsafe {
        match (a.tag, b.tag) {
            (TAG_INT, TAG_INT) => {
                let av = a.data.int_val;
                let bv = b.data.int_val;
                if bv == 0 {
                    TokValue::nil()
                } else if av == i64::MIN && bv == -1 {
                    TokValue::from_int(i64::MIN) // wrapping: matches codegen
                } else {
                    TokValue::from_int(av / bv)
                }
            }
            (TAG_FLOAT, TAG_FLOAT) => TokValue::from_float(a.data.float_val / b.data.float_val),
            (TAG_INT, TAG_FLOAT) => TokValue::from_float(a.data.int_val as f64 / b.data.float_val),
            (TAG_FLOAT, TAG_INT) => TokValue::from_float(a.data.float_val / b.data.int_val as f64),
            _ => TokValue::nil(),
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_value_mod(a: TokValue, b: TokValue) -> TokValue {
    unsafe {
        match (a.tag, b.tag) {
            (TAG_INT, TAG_INT) => {
                let av = a.data.int_val;
                let bv = b.data.int_val;
                if bv == 0 {
                    TokValue::nil()
                } else if av == i64::MIN && bv == -1 {
                    TokValue::from_int(0) // i64::MIN % -1 = 0
                } else {
                    TokValue::from_int(av % bv)
                }
            }
            (TAG_FLOAT, TAG_FLOAT) => TokValue::from_float(a.data.float_val % b.data.float_val),
            (TAG_INT, TAG_FLOAT) => TokValue::from_float(a.data.int_val as f64 % b.data.float_val),
            (TAG_FLOAT, TAG_INT) => TokValue::from_float(a.data.float_val % b.data.int_val as f64),
            _ => TokValue::nil(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Power functions (for ** operator)
// ═══════════════════════════════════════════════════════════════

/// Float power: base.powf(exp)
#[no_mangle]
pub extern "C" fn tok_pow_f64(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Integer power: base^exp using exponentiation by squaring.
#[no_mangle]
pub extern "C" fn tok_pow_int(base: i64, exp: i64) -> i64 {
    if exp < 0 {
        return 0; // Integer power with negative exponent → 0 (truncated)
    }
    let mut result: i64 = 1;
    let mut b = base;
    let mut e = exp as u64;
    while e > 0 {
        if e & 1 == 1 {
            result = result.wrapping_mul(b);
        }
        b = b.wrapping_mul(b);
        e >>= 1;
    }
    result
}

/// Dynamic-dispatch power for Any-typed operands.
#[no_mangle]
pub extern "C" fn tok_value_pow(a: TokValue, b: TokValue) -> TokValue {
    unsafe {
        match (a.tag, b.tag) {
            (TAG_INT, TAG_INT) => TokValue::from_int(tok_pow_int(a.data.int_val, b.data.int_val)),
            (TAG_FLOAT, TAG_FLOAT) => TokValue::from_float(a.data.float_val.powf(b.data.float_val)),
            (TAG_INT, TAG_FLOAT) => {
                TokValue::from_float((a.data.int_val as f64).powf(b.data.float_val))
            }
            (TAG_FLOAT, TAG_INT) => {
                TokValue::from_float(a.data.float_val.powf(b.data.int_val as f64))
            }
            _ => TokValue::nil(),
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_value_eq(a: TokValue, b: TokValue) -> i8 {
    if a == b {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn tok_value_lt(a: TokValue, b: TokValue) -> i8 {
    unsafe {
        match (a.tag, b.tag) {
            (TAG_INT, TAG_INT) => {
                if a.data.int_val < b.data.int_val {
                    1
                } else {
                    0
                }
            }
            (TAG_FLOAT, TAG_FLOAT) => {
                if a.data.float_val < b.data.float_val {
                    1
                } else {
                    0
                }
            }
            (TAG_INT, TAG_FLOAT) => {
                if (a.data.int_val as f64) < b.data.float_val {
                    1
                } else {
                    0
                }
            }
            (TAG_FLOAT, TAG_INT) => {
                if a.data.float_val < (b.data.int_val as f64) {
                    1
                } else {
                    0
                }
            }
            (TAG_STRING, TAG_STRING) => {
                if a.data.string_ptr.is_null() || b.data.string_ptr.is_null() {
                    0
                } else if (*a.data.string_ptr).data < (*b.data.string_ptr).data {
                    1
                } else {
                    0
                }
            }
            _ => 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_value_truthiness(a: TokValue) -> i8 {
    if a.truthiness() {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn tok_value_negate(a: TokValue) -> TokValue {
    unsafe {
        match a.tag {
            TAG_INT => TokValue::from_int(a.data.int_val.wrapping_neg()),
            TAG_FLOAT => TokValue::from_float(-a.data.float_val),
            _ => TokValue::nil(),
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_value_not(a: TokValue) -> i8 {
    if a.truthiness() {
        0
    } else {
        1
    }
}

// ═══════════════════════════════════════════════════════════════
// Type check: is(value, type_string)
// ═══════════════════════════════════════════════════════════════

/// is(val, type_str) -> Bool
/// Checks if the value's type matches the given type string.
#[no_mangle]
pub extern "C" fn tok_is(val: TokValue, type_str: *mut TokString) -> i8 {
    if type_str.is_null() {
        return 0;
    }
    let expected = unsafe { &(*type_str).data };
    if val.type_name() == expected.as_str() {
        1
    } else {
        0
    }
}

// ═══════════════════════════════════════════════════════════════
// Array: pop, freq, zip
// ═══════════════════════════════════════════════════════════════

/// pop(arr) -> (shortened_array, last_element)
/// Returns a tuple of (array without last element, last element).
#[no_mangle]
pub extern "C" fn tok_array_pop(arr: *mut crate::array::TokArray) -> TokValue {
    null_check!(arr, "tok_array_pop: null array");
    unsafe {
        let src = &(*arr).data;
        if src.is_empty() {
            // Return ([], N)
            let new_arr = crate::array::TokArray::alloc();
            let elems = vec![TokValue::from_array(new_arr), TokValue::nil()];
            return TokValue::from_tuple(crate::tuple::TokTuple::alloc(elems));
        }
        let last = src[src.len() - 1];
        last.rc_inc();
        let new_arr = crate::array::TokArray::alloc();
        for item in &src[..src.len() - 1] {
            item.rc_inc();
            (*new_arr).data.push(*item);
        }
        let elems = vec![TokValue::from_array(new_arr), last];
        TokValue::from_tuple(crate::tuple::TokTuple::alloc(elems))
    }
}

/// freq(arr) -> Map  {element: count}
/// Counts occurrences of each element. Keys are stringified values.
#[no_mangle]
pub extern "C" fn tok_array_freq(arr: *mut crate::array::TokArray) -> *mut crate::map::TokMap {
    null_check!(arr, "tok_array_freq: null array");
    unsafe {
        let m = crate::map::TokMap::alloc();
        for val in &(*arr).data {
            let key = format!("{}", val);
            if let Some(existing) = (*m).data.get_mut(&key) {
                // Increment count
                if existing.tag == TAG_INT {
                    *existing = TokValue::from_int(existing.data.int_val + 1);
                }
            } else {
                (*m).data.insert(key, TokValue::from_int(1));
            }
        }
        m
    }
}

/// zip(a, b) -> Array<Tuple>
/// Zips two arrays into an array of tuples.
#[no_mangle]
pub extern "C" fn tok_array_zip(
    a: *mut crate::array::TokArray,
    b: *mut crate::array::TokArray,
) -> *mut crate::array::TokArray {
    null_check!(a, "tok_array_zip: null array a");
    null_check!(b, "tok_array_zip: null array b");
    unsafe {
        let result = crate::array::TokArray::alloc();
        let len = std::cmp::min((*a).data.len(), (*b).data.len());
        for i in 0..len {
            let va = (*a).data[i];
            let vb = (*b).data[i];
            va.rc_inc();
            vb.rc_inc();
            let tup = crate::tuple::TokTuple::alloc(vec![va, vb]);
            (*result).data.push(TokValue::from_tuple(tup));
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════
// Map: top(m, n)
// ═══════════════════════════════════════════════════════════════

/// top(map, n) -> Array<Tuple>
/// Returns the top N entries by value (descending), as array of (key, value) tuples.
#[no_mangle]
pub extern "C" fn tok_map_top(m: *mut crate::map::TokMap, n: i64) -> *mut crate::array::TokArray {
    null_check!(m, "tok_map_top: null map");
    unsafe {
        let mut entries: Vec<(&String, &TokValue)> = (*m).data.iter().collect();
        // Sort by value descending
        entries.sort_by(|a, b| {
            let va = match a.1.tag {
                TAG_INT => a.1.data.int_val as f64,
                TAG_FLOAT => a.1.data.float_val,
                _ => 0.0,
            };
            let vb = match b.1.tag {
                TAG_INT => b.1.data.int_val as f64,
                TAG_FLOAT => b.1.data.float_val,
                _ => 0.0,
            };
            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
        });
        let result = crate::array::TokArray::alloc();
        let take = std::cmp::min(n as usize, entries.len());
        for (k, v) in entries.into_iter().take(take) {
            v.rc_inc();
            let tup = crate::tuple::TokTuple::alloc(vec![
                TokValue::from_string(TokString::alloc(k.clone())),
                *v,
            ]);
            (*result).data.push(TokValue::from_tuple(tup));
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════
// OS builtins promoted to core: args(), env(k)
// ═══════════════════════════════════════════════════════════════

/// args() -> Array<Str>
/// Returns command-line arguments (excluding the program name).
#[no_mangle]
pub extern "C" fn tok_args() -> *mut crate::array::TokArray {
    let arr = crate::array::TokArray::alloc();
    // Skip the first arg (program name) to match typical scripting language behavior
    for arg in std::env::args().skip(1) {
        let s = TokString::alloc(arg);
        unsafe {
            (*arr).data.push(TokValue::from_string(s));
        }
    }
    arr
}

/// env(name) -> Str | Nil
/// Returns the value of an environment variable, or Nil if not set.
#[no_mangle]
pub extern "C" fn tok_env(name: *mut TokString) -> TokValue {
    if name.is_null() {
        return TokValue::nil();
    }
    unsafe {
        match std::env::var(&(*name).data) {
            Ok(val) => TokValue::from_string(TokString::alloc(val)),
            Err(_) => TokValue::nil(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rc_inc_dec() {
        let s = TokString::alloc("test".to_string());
        tok_rc_inc(s as *mut u8);
        unsafe {
            assert_eq!((*s).rc.load(std::sync::atomic::Ordering::Relaxed), 2);
        }
        assert_eq!(tok_rc_dec(s as *mut u8), 0); // 2 -> 1
        assert_eq!(tok_rc_dec(s as *mut u8), 1); // 1 -> 0
        unsafe {
            drop(Box::from_raw(s));
        }
    }

    #[test]
    fn test_rc_null() {
        tok_rc_inc(std::ptr::null_mut());
        assert_eq!(tok_rc_dec(std::ptr::null_mut()), 0);
    }

    #[test]
    fn test_clock() {
        let t = tok_clock();
        assert!(t > 0);
    }

    #[test]
    fn test_abs_int() {
        assert_eq!(tok_abs_int(-5), 5);
        assert_eq!(tok_abs_int(5), 5);
        assert_eq!(tok_abs_int(0), 0);
    }

    #[test]
    fn test_abs_float() {
        assert!((tok_abs_float(-3.14) - 3.14).abs() < 1e-10);
        assert!((tok_abs_float(3.14) - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_floor_ceil() {
        assert_eq!(tok_floor(3.7), 3);
        assert_eq!(tok_floor(3.2), 3);
        assert_eq!(tok_floor(-1.5), -2);
        assert_eq!(tok_ceil(3.2), 4);
        assert_eq!(tok_ceil(3.0), 3);
        assert_eq!(tok_ceil(-1.5), -1);
    }

    #[test]
    fn test_rand() {
        let r = tok_rand();
        assert!(r >= 0.0 && r < 1.0);
        // Check it produces different values
        let r2 = tok_rand();
        // They could theoretically be equal, but that's astronomically unlikely
        assert_ne!(r, r2);
    }

    #[test]
    fn test_to_int() {
        assert_eq!(tok_to_int(TokValue::from_int(42)), 42);
        assert_eq!(tok_to_int(TokValue::from_float(3.7)), 3);
        assert_eq!(tok_to_int(TokValue::from_bool(true)), 1);
        assert_eq!(tok_to_int(TokValue::from_bool(false)), 0);

        let s = TokString::alloc("123".to_string());
        assert_eq!(tok_to_int(TokValue::from_string(s)), 123);
        unsafe {
            drop(Box::from_raw(s));
        }

        assert_eq!(tok_to_int(TokValue::nil()), 0);
    }

    #[test]
    fn test_to_float() {
        assert!((tok_to_float(TokValue::from_int(42)) - 42.0).abs() < 1e-10);
        assert!((tok_to_float(TokValue::from_float(3.14)) - 3.14).abs() < 1e-10);

        let s = TokString::alloc("3.14".to_string());
        assert!((tok_to_float(TokValue::from_string(s)) - 3.14).abs() < 1e-10);
        unsafe {
            drop(Box::from_raw(s));
        }
    }

    #[test]
    fn test_value_add_int() {
        let result = tok_value_add(TokValue::from_int(3), TokValue::from_int(4));
        assert_eq!(result.tag, TAG_INT);
        unsafe {
            assert_eq!(result.data.int_val, 7);
        }
    }

    #[test]
    fn test_value_add_float() {
        let result = tok_value_add(TokValue::from_float(1.5), TokValue::from_float(2.5));
        assert_eq!(result.tag, TAG_FLOAT);
        unsafe {
            assert!((result.data.float_val - 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_value_add_mixed() {
        let result = tok_value_add(TokValue::from_int(1), TokValue::from_float(2.5));
        assert_eq!(result.tag, TAG_FLOAT);
        unsafe {
            assert!((result.data.float_val - 3.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_value_add_string() {
        let a = TokString::alloc("hello ".to_string());
        let b = TokString::alloc("world".to_string());
        let result = tok_value_add(TokValue::from_string(a), TokValue::from_string(b));
        assert_eq!(result.tag, TAG_STRING);
        unsafe {
            assert_eq!(&(*result.data.string_ptr).data, "hello world");
            // COW may reuse `a`, so result.data.string_ptr may equal a.
            // Only free distinct pointers to avoid double-free.
            if result.data.string_ptr != a {
                drop(Box::from_raw(a));
            }
            drop(Box::from_raw(b));
            drop(Box::from_raw(result.data.string_ptr));
        }
    }

    #[test]
    fn test_value_sub() {
        let result = tok_value_sub(TokValue::from_int(10), TokValue::from_int(3));
        unsafe {
            assert_eq!(result.data.int_val, 7);
        }
    }

    #[test]
    fn test_value_mul() {
        let result = tok_value_mul(TokValue::from_int(3), TokValue::from_int(4));
        unsafe {
            assert_eq!(result.data.int_val, 12);
        }
    }

    #[test]
    fn test_value_div() {
        let result = tok_value_div(TokValue::from_int(10), TokValue::from_int(3));
        unsafe {
            assert_eq!(result.data.int_val, 3);
        }

        // Division by zero
        let result = tok_value_div(TokValue::from_int(10), TokValue::from_int(0));
        assert_eq!(result.tag, 0); // NIL
    }

    #[test]
    fn test_value_mod() {
        let result = tok_value_mod(TokValue::from_int(10), TokValue::from_int(3));
        unsafe {
            assert_eq!(result.data.int_val, 1);
        }
    }

    #[test]
    fn test_value_eq() {
        assert_eq!(
            tok_value_eq(TokValue::from_int(1), TokValue::from_int(1)),
            1
        );
        assert_eq!(
            tok_value_eq(TokValue::from_int(1), TokValue::from_int(2)),
            0
        );
        assert_eq!(tok_value_eq(TokValue::nil(), TokValue::nil()), 1);
    }

    #[test]
    fn test_value_lt() {
        assert_eq!(
            tok_value_lt(TokValue::from_int(1), TokValue::from_int(2)),
            1
        );
        assert_eq!(
            tok_value_lt(TokValue::from_int(2), TokValue::from_int(1)),
            0
        );
        assert_eq!(
            tok_value_lt(TokValue::from_int(1), TokValue::from_int(1)),
            0
        );
    }

    #[test]
    fn test_value_truthiness() {
        assert_eq!(tok_value_truthiness(TokValue::from_int(1)), 1);
        assert_eq!(tok_value_truthiness(TokValue::from_int(0)), 0);
        assert_eq!(tok_value_truthiness(TokValue::nil()), 0);
        assert_eq!(tok_value_truthiness(TokValue::from_bool(true)), 1);
        assert_eq!(tok_value_truthiness(TokValue::from_bool(false)), 0);
    }

    #[test]
    fn test_value_negate() {
        let result = tok_value_negate(TokValue::from_int(5));
        unsafe {
            assert_eq!(result.data.int_val, -5);
        }

        let result = tok_value_negate(TokValue::from_float(3.14));
        unsafe {
            assert!((result.data.float_val - (-3.14)).abs() < 1e-10);
        }

        // Negate nil → nil
        let result = tok_value_negate(TokValue::nil());
        assert_eq!(result.tag, 0);
    }

    #[test]
    fn test_value_not() {
        assert_eq!(tok_value_not(TokValue::from_bool(true)), 0);
        assert_eq!(tok_value_not(TokValue::from_bool(false)), 1);
        assert_eq!(tok_value_not(TokValue::nil()), 1);
        assert_eq!(tok_value_not(TokValue::from_int(1)), 0);
    }

    #[test]
    fn test_type_of() {
        let s = tok_type_of(TokValue::from_int(42));
        unsafe {
            assert_eq!(&(*s).data, "int");
            drop(Box::from_raw(s));
        }

        let s = tok_type_of(TokValue::nil());
        unsafe {
            assert_eq!(&(*s).data, "nil");
            drop(Box::from_raw(s));
        }
    }

    #[test]
    fn test_int_to_string() {
        let s = tok_int_to_string(42);
        unsafe {
            assert_eq!(&(*s).data, "42");
            drop(Box::from_raw(s));
        }
    }

    #[test]
    fn test_float_to_string() {
        let s = tok_float_to_string(3.14);
        unsafe {
            assert_eq!(&(*s).data, "3.14");
            drop(Box::from_raw(s));
        }
    }

    #[test]
    fn test_bool_to_string() {
        let s = tok_bool_to_string(1);
        unsafe {
            assert_eq!(&(*s).data, "T");
            drop(Box::from_raw(s));
        }

        let s = tok_bool_to_string(0);
        unsafe {
            assert_eq!(&(*s).data, "F");
            drop(Box::from_raw(s));
        }
    }

    #[test]
    fn test_value_to_string() {
        let s = tok_value_to_string(TokValue::from_int(42));
        unsafe {
            assert_eq!(&(*s).data, "42");
            drop(Box::from_raw(s));
        }

        let s = tok_value_to_string(TokValue::nil());
        unsafe {
            assert_eq!(&(*s).data, "N");
            drop(Box::from_raw(s));
        }
    }

    #[test]
    fn test_print_capture() {
        // We can't easily capture stdout in a unit test without extra deps,
        // so just verify the functions don't crash.
        tok_print(TokValue::from_int(42));
        tok_println(TokValue::from_int(42));
        tok_print_int(42);
        tok_println_int(42);
        tok_print_float(3.14);
        tok_println_float(3.14);
        tok_print_bool(1);
        tok_println_bool(0);
        let s = TokString::alloc("test".to_string());
        tok_print_string(s);
        tok_println_string(s);
        tok_print_string(std::ptr::null_mut());
        tok_println_string(std::ptr::null_mut());
        unsafe {
            drop(Box::from_raw(s));
        }
    }
}
