//! Reference-counted dynamic array for the Tok runtime.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{fence, AtomicU32, Ordering};

use crate::closure::TokClosure;
use crate::string::TokString;
use crate::value::{TokValue, TAG_ARRAY, TAG_BOOL, TAG_FLOAT, TAG_INT, TAG_NIL, TAG_STRING};

// ═══════════════════════════════════════════════════════════════
// HashableTokValue — wrapper for dedup via HashSet
// ═══════════════════════════════════════════════════════════════

/// Wrapper around TokValue that provides Hash + Eq for use in HashSet.
/// Floats use bit-level comparison (matching tok_values_equal semantics).
/// Non-primitive types fall back to raw pointer identity.
struct HashableTokValue(TokValue);

impl Hash for HashableTokValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let v = &self.0;
        v.tag.hash(state);
        unsafe {
            match v.tag {
                TAG_NIL => {}
                TAG_INT => v.data.int_val.hash(state),
                TAG_FLOAT => v.data.float_val.to_bits().hash(state),
                3 => v.data.bool_val.hash(state), // BOOL
                TAG_STRING => {
                    if !v.data.string_ptr.is_null() {
                        (*v.data.string_ptr).data.hash(state);
                    }
                }
                _ => v.data._raw.hash(state),
            }
        }
    }
}

impl PartialEq for HashableTokValue {
    fn eq(&self, other: &Self) -> bool {
        tok_values_equal(&self.0, &other.0)
    }
}

impl Eq for HashableTokValue {}

// ═══════════════════════════════════════════════════════════════
// TagData — return type for filter/reduce closure calls
// ═══════════════════════════════════════════════════════════════

/// A (tag, data) pair returned by closure calls.
/// Must match the Cranelift calling convention: two i64 return values.
#[repr(C)]
pub struct TagData {
    pub tag: i64,
    pub data: i64,
}

// ═══════════════════════════════════════════════════════════════
// TokArray
// ═══════════════════════════════════════════════════════════════

#[repr(C)]
pub struct TokArray {
    pub rc: AtomicU32,
    pub data: Vec<TokValue>,
}

impl Default for TokArray {
    fn default() -> Self {
        Self::new()
    }
}

impl TokArray {
    pub fn new() -> Self {
        TokArray {
            rc: AtomicU32::new(1),
            data: Vec::new(),
        }
    }

    pub fn rc_inc(&self) {
        self.rc.fetch_add(1, Ordering::Relaxed);
    }

    pub fn rc_dec(&self) -> bool {
        if self.rc.fetch_sub(1, Ordering::Release) == 1 {
            fence(Ordering::Acquire);
            true
        } else {
            false
        }
    }

    pub fn alloc() -> *mut TokArray {
        Box::into_raw(Box::new(TokArray::new()))
    }
}

// ═══════════════════════════════════════════════════════════════
// extern "C" API
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_array_alloc() -> *mut TokArray {
    TokArray::alloc()
}

#[no_mangle]
pub extern "C" fn tok_array_push(arr: *mut TokArray, val: TokValue) -> *mut TokArray {
    assert!(!arr.is_null(), "tok_array_push: null array");
    unsafe {
        val.rc_inc();
        (*arr).data.push(val);
    }
    arr
}

#[no_mangle]
pub extern "C" fn tok_array_get(arr: *mut TokArray, idx: i64) -> TokValue {
    assert!(!arr.is_null(), "tok_array_get: null array");
    unsafe {
        let len = (*arr).data.len() as i64;
        let real_idx = if idx < 0 { len + idx } else { idx };
        if real_idx < 0 || real_idx >= len {
            TokValue::nil()
        } else {
            let val = (*arr).data[real_idx as usize];
            val.rc_inc();
            val
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_array_set(arr: *mut TokArray, idx: i64, val: TokValue) {
    assert!(!arr.is_null(), "tok_array_set: null array");
    unsafe {
        let len = (*arr).data.len() as i64;
        let real_idx = if idx < 0 { len + idx } else { idx };
        if real_idx >= 0 && real_idx < len {
            let old = (*arr).data[real_idx as usize];
            old.rc_dec();
            val.rc_inc();
            (*arr).data[real_idx as usize] = val;
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_array_len(arr: *mut TokArray) -> i64 {
    assert!(!arr.is_null(), "tok_array_len: null array");
    unsafe { (*arr).data.len() as i64 }
}

#[no_mangle]
pub extern "C" fn tok_array_slice(arr: *mut TokArray, start: i64, end: i64) -> *mut TokArray {
    assert!(!arr.is_null(), "tok_array_slice: null array");
    unsafe {
        let len = (*arr).data.len() as i64;
        let s = start.max(0).min(len) as usize;
        let e = end.max(0).min(len) as usize;
        let result = TokArray::alloc();
        if s < e {
            for v in &(*arr).data[s..e] {
                v.rc_inc();
                (*result).data.push(*v);
            }
        }
        result
    }
}

#[no_mangle]
pub extern "C" fn tok_array_sort(arr: *mut TokArray) -> *mut TokArray {
    assert!(!arr.is_null(), "tok_array_sort: null array");
    unsafe {
        let result = TokArray::alloc();
        let mut items: Vec<TokValue> = (*arr).data.clone();
        items.sort_by(tok_value_compare);
        for v in &items {
            v.rc_inc();
        }
        (*result).data = items;
        result
    }
}

#[no_mangle]
pub extern "C" fn tok_array_rev(arr: *mut TokArray) -> *mut TokArray {
    assert!(!arr.is_null(), "tok_array_rev: null array");
    unsafe {
        let result = TokArray::alloc();
        let mut items: Vec<TokValue> = (*arr).data.clone();
        items.reverse();
        for v in &items {
            v.rc_inc();
        }
        (*result).data = items;
        result
    }
}

#[no_mangle]
pub extern "C" fn tok_array_flat(arr: *mut TokArray) -> *mut TokArray {
    assert!(!arr.is_null(), "tok_array_flat: null array");
    unsafe {
        let result = TokArray::alloc();
        flatten_into(&(*arr).data, &mut (*result).data);
        result
    }
}

unsafe fn flatten_into(src: &[TokValue], dst: &mut Vec<TokValue>) {
    for v in src {
        if v.tag == TAG_ARRAY && !v.data.array_ptr.is_null() {
            flatten_into(&(*v.data.array_ptr).data, dst);
        } else {
            v.rc_inc();
            dst.push(*v);
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_array_uniq(arr: *mut TokArray) -> *mut TokArray {
    assert!(!arr.is_null(), "tok_array_uniq: null array");
    unsafe {
        let result = TokArray::alloc();
        let mut seen = HashSet::with_capacity((*arr).data.len());
        for v in &(*arr).data {
            if seen.insert(HashableTokValue(*v)) {
                v.rc_inc();
                (*result).data.push(*v);
            }
        }
        result
    }
}

#[no_mangle]
pub extern "C" fn tok_array_join(arr: *mut TokArray, sep: *mut TokString) -> *mut TokString {
    assert!(!arr.is_null(), "tok_array_join: null array");
    assert!(!sep.is_null(), "tok_array_join: null separator");
    unsafe {
        let sep_str = &(*sep).data;
        let data = &(*arr).data;
        if data.is_empty() {
            return TokString::alloc(String::new());
        }
        // Direct-write: build result string without intermediate Vec<String>.
        // Pre-estimate capacity for string-only arrays to avoid reallocation.
        let mut est_len = sep_str.len() * (data.len() - 1);
        for v in data.iter() {
            if v.tag == TAG_STRING && !v.data.string_ptr.is_null() {
                est_len += (*v.data.string_ptr).data.len();
            } else {
                est_len += 16; // rough estimate for non-string values
            }
        }
        let mut result = String::with_capacity(est_len);
        for (i, v) in data.iter().enumerate() {
            if i > 0 {
                result.push_str(sep_str);
            }
            // Fast path: string elements — no format! overhead
            if v.tag == TAG_STRING && !v.data.string_ptr.is_null() {
                result.push_str(&(*v.data.string_ptr).data);
            } else {
                use std::fmt::Write;
                let _ = write!(result, "{}", v);
            }
        }
        TokString::alloc(result)
    }
}

#[no_mangle]
pub extern "C" fn tok_array_concat(a: *mut TokArray, b: *mut TokArray) -> *mut TokArray {
    assert!(!a.is_null(), "tok_array_concat: null lhs");
    assert!(!b.is_null(), "tok_array_concat: null rhs");
    unsafe {
        let result = TokArray::alloc();
        for v in &(*a).data {
            v.rc_inc();
            (*result).data.push(*v);
        }
        for v in &(*b).data {
            v.rc_inc();
            (*result).data.push(*v);
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════
// Array numeric builtins
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_array_min(arr: *mut TokArray) -> TokValue {
    assert!(!arr.is_null(), "tok_array_min: null array");
    unsafe {
        if (*arr).data.is_empty() {
            return TokValue::nil();
        }
        let mut min = (*arr).data[0];
        for v in &(*arr).data[1..] {
            if tok_value_compare(v, &min) == std::cmp::Ordering::Less {
                min = *v;
            }
        }
        min.rc_inc();
        min
    }
}

#[no_mangle]
pub extern "C" fn tok_array_max(arr: *mut TokArray) -> TokValue {
    assert!(!arr.is_null(), "tok_array_max: null array");
    unsafe {
        if (*arr).data.is_empty() {
            return TokValue::nil();
        }
        let mut max = (*arr).data[0];
        for v in &(*arr).data[1..] {
            if tok_value_compare(v, &max) == std::cmp::Ordering::Greater {
                max = *v;
            }
        }
        max.rc_inc();
        max
    }
}

#[no_mangle]
pub extern "C" fn tok_array_sum(arr: *mut TokArray) -> TokValue {
    assert!(!arr.is_null(), "tok_array_sum: null array");
    unsafe {
        if (*arr).data.is_empty() {
            return TokValue::from_int(0);
        }
        let mut has_float = false;
        let mut int_sum: i64 = 0;
        let mut float_sum: f64 = 0.0;
        for v in &(*arr).data {
            match v.tag {
                TAG_INT => {
                    int_sum += v.data.int_val;
                    float_sum += v.data.int_val as f64;
                }
                TAG_FLOAT => {
                    has_float = true;
                    float_sum += v.data.float_val;
                }
                _ => {}
            }
        }
        if has_float {
            TokValue::from_float(float_sum)
        } else {
            TokValue::from_int(int_sum)
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Filter and Reduce (closure-based)
// ═══════════════════════════════════════════════════════════════

/// Filter array elements using a closure predicate.
/// Closure signature: (env: *mut u8, tag: i64, data: i64) -> TagData
#[no_mangle]
pub extern "C" fn tok_array_filter(arr: *mut TokArray, closure: *mut TokClosure) -> *mut TokArray {
    assert!(!arr.is_null(), "tok_array_filter: null array");
    assert!(!closure.is_null(), "tok_array_filter: null closure");
    unsafe {
        let fn_ptr: extern "C" fn(*mut u8, i64, i64) -> TagData =
            std::mem::transmute((*closure).fn_ptr);
        let env = (*closure).env_ptr;
        let result = TokArray::alloc();
        for elem in &(*arr).data {
            let td = fn_ptr(env, elem.tag as i64, elem.data._raw as i64);
            // Check truthiness of result
            let is_truthy = match td.tag as u8 {
                TAG_BOOL | TAG_INT => td.data != 0,
                TAG_NIL => false,
                _ => true,
            };
            if is_truthy {
                elem.rc_inc();
                (*result).data.push(*elem);
            }
        }
        result
    }
}

/// Reduce array elements using a closure accumulator.
/// Closure signature: (env: *mut u8, acc_tag: i64, acc_data: i64, elem_tag: i64, elem_data: i64) -> TagData
#[no_mangle]
pub extern "C" fn tok_array_reduce(
    arr: *mut TokArray,
    init_tag: i64,
    init_data: i64,
    closure: *mut TokClosure,
) -> TagData {
    assert!(!arr.is_null(), "tok_array_reduce: null array");
    assert!(!closure.is_null(), "tok_array_reduce: null closure");
    unsafe {
        let fn_ptr: extern "C" fn(*mut u8, i64, i64, i64, i64) -> TagData =
            std::mem::transmute((*closure).fn_ptr);
        let env = (*closure).env_ptr;
        let data = &(*arr).data;

        // If init is Nil (tag=0, data=0), use first element as init
        let (mut acc_tag, mut acc_data, start_idx) =
            if init_tag == 0 && init_data == 0 && !data.is_empty() {
                (data[0].tag as i64, data[0].data._raw as i64, 1)
            } else {
                (init_tag, init_data, 0)
            };

        for elem in &data[start_idx..] {
            let result = fn_ptr(
                env,
                acc_tag,
                acc_data,
                elem.tag as i64,
                elem.data._raw as i64,
            );
            acc_tag = result.tag;
            acc_data = result.data;
        }

        TagData {
            tag: acc_tag,
            data: acc_data,
        }
    }
}

/// Parallel map: apply a closure to each array element in a separate thread.
/// Returns a new array with results in the same order.
/// Closure signature: (env: *mut u8, tag: i64, data: i64) -> TagData
#[no_mangle]
pub extern "C" fn tok_pmap(arr: *mut TokArray, closure: *mut TokClosure) -> *mut TokArray {
    assert!(!arr.is_null(), "tok_pmap: null array");
    assert!(!closure.is_null(), "tok_pmap: null closure");
    unsafe {
        let fn_ptr: extern "C" fn(*mut u8, i64, i64) -> TagData =
            std::mem::transmute((*closure).fn_ptr);
        let env = (*closure).env_ptr;
        let data = &(*arr).data;

        if data.is_empty() {
            return TokArray::alloc();
        }

        // Spawn one thread per element
        let fn_ptr_usize = fn_ptr as usize;
        let env_usize = env as usize;
        let handles: Vec<_> = data
            .iter()
            .map(|elem| {
                let tag = elem.tag as i64;
                let data = elem.data._raw as i64;
                std::thread::spawn(move || {
                    let fp: extern "C" fn(*mut u8, i64, i64) -> TagData =
                        std::mem::transmute(fn_ptr_usize);
                    let ep = env_usize as *mut u8;
                    fp(ep, tag, data)
                })
            })
            .collect();

        // Join in order, collect results
        let result = TokArray::alloc();
        for h in handles {
            match h.join() {
                Ok(td) => {
                    let val = TokValue::from_tag_data(td.tag, td.data);
                    (*result).data.push(val);
                }
                Err(_) => {
                    (*result).data.push(TokValue::nil());
                }
            }
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

pub fn tok_value_compare(a: &TokValue, b: &TokValue) -> std::cmp::Ordering {
    unsafe {
        match (a.tag, b.tag) {
            (TAG_INT, TAG_INT) => a.data.int_val.cmp(&b.data.int_val),
            (TAG_FLOAT, TAG_FLOAT) => a
                .data
                .float_val
                .partial_cmp(&b.data.float_val)
                .unwrap_or(std::cmp::Ordering::Equal),
            (TAG_INT, TAG_FLOAT) => (a.data.int_val as f64)
                .partial_cmp(&b.data.float_val)
                .unwrap_or(std::cmp::Ordering::Equal),
            (TAG_FLOAT, TAG_INT) => a
                .data
                .float_val
                .partial_cmp(&(b.data.int_val as f64))
                .unwrap_or(std::cmp::Ordering::Equal),
            (TAG_STRING, TAG_STRING) => {
                if a.data.string_ptr.is_null() || b.data.string_ptr.is_null() {
                    std::cmp::Ordering::Equal
                } else {
                    (*a.data.string_ptr).data.cmp(&(*b.data.string_ptr).data)
                }
            }
            _ => a.tag.cmp(&b.tag),
        }
    }
}

pub fn tok_values_equal(a: &TokValue, b: &TokValue) -> bool {
    if a.tag != b.tag {
        return false;
    }
    unsafe {
        match a.tag {
            0 => true, // NIL
            TAG_INT => a.data.int_val == b.data.int_val,
            TAG_FLOAT => a.data.float_val == b.data.float_val,
            3 => a.data.bool_val == b.data.bool_val, // BOOL
            TAG_STRING => {
                if a.data.string_ptr.is_null() && b.data.string_ptr.is_null() {
                    true
                } else if a.data.string_ptr.is_null() || b.data.string_ptr.is_null() {
                    false
                } else {
                    (*a.data.string_ptr).data == (*b.data.string_ptr).data
                }
            }
            _ => a.data._raw == b.data._raw,
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
    fn test_alloc_empty() {
        let arr = tok_array_alloc();
        assert_eq!(tok_array_len(arr), 0);
        unsafe {
            drop(Box::from_raw(arr));
        }
    }

    #[test]
    fn test_push_and_get() {
        let arr = tok_array_alloc();
        tok_array_push(arr, TokValue::from_int(10));
        tok_array_push(arr, TokValue::from_int(20));
        tok_array_push(arr, TokValue::from_int(30));
        assert_eq!(tok_array_len(arr), 3);

        let v0 = tok_array_get(arr, 0);
        let v1 = tok_array_get(arr, 1);
        let v2 = tok_array_get(arr, 2);
        unsafe {
            assert_eq!(v0.data.int_val, 10);
            assert_eq!(v1.data.int_val, 20);
            assert_eq!(v2.data.int_val, 30);
        }

        // Out of bounds
        let vn = tok_array_get(arr, 5);
        assert_eq!(vn.tag, 0); // NIL

        // Negative indexing
        let v_last = tok_array_get(arr, -1);
        unsafe {
            assert_eq!(v_last.data.int_val, 30);
        }

        unsafe {
            drop(Box::from_raw(arr));
        }
    }

    #[test]
    fn test_set() {
        let arr = tok_array_alloc();
        tok_array_push(arr, TokValue::from_int(10));
        tok_array_set(arr, 0, TokValue::from_int(99));
        let v = tok_array_get(arr, 0);
        unsafe {
            assert_eq!(v.data.int_val, 99);
            drop(Box::from_raw(arr));
        }
    }

    #[test]
    fn test_slice() {
        let arr = tok_array_alloc();
        for i in 0..5 {
            tok_array_push(arr, TokValue::from_int(i));
        }
        let sliced = tok_array_slice(arr, 1, 4);
        assert_eq!(tok_array_len(sliced), 3);
        unsafe {
            assert_eq!((*sliced).data[0].data.int_val, 1);
            assert_eq!((*sliced).data[1].data.int_val, 2);
            assert_eq!((*sliced).data[2].data.int_val, 3);
            drop(Box::from_raw(arr));
            drop(Box::from_raw(sliced));
        }
    }

    #[test]
    fn test_sort() {
        let arr = tok_array_alloc();
        tok_array_push(arr, TokValue::from_int(3));
        tok_array_push(arr, TokValue::from_int(1));
        tok_array_push(arr, TokValue::from_int(2));
        let sorted = tok_array_sort(arr);
        unsafe {
            assert_eq!((*sorted).data[0].data.int_val, 1);
            assert_eq!((*sorted).data[1].data.int_val, 2);
            assert_eq!((*sorted).data[2].data.int_val, 3);
            drop(Box::from_raw(arr));
            drop(Box::from_raw(sorted));
        }
    }

    #[test]
    fn test_rev() {
        let arr = tok_array_alloc();
        tok_array_push(arr, TokValue::from_int(1));
        tok_array_push(arr, TokValue::from_int(2));
        tok_array_push(arr, TokValue::from_int(3));
        let reversed = tok_array_rev(arr);
        unsafe {
            assert_eq!((*reversed).data[0].data.int_val, 3);
            assert_eq!((*reversed).data[1].data.int_val, 2);
            assert_eq!((*reversed).data[2].data.int_val, 1);
            drop(Box::from_raw(arr));
            drop(Box::from_raw(reversed));
        }
    }

    #[test]
    fn test_concat() {
        let a = tok_array_alloc();
        tok_array_push(a, TokValue::from_int(1));
        tok_array_push(a, TokValue::from_int(2));
        let b = tok_array_alloc();
        tok_array_push(b, TokValue::from_int(3));
        tok_array_push(b, TokValue::from_int(4));
        let c = tok_array_concat(a, b);
        assert_eq!(tok_array_len(c), 4);
        unsafe {
            assert_eq!((*c).data[0].data.int_val, 1);
            assert_eq!((*c).data[3].data.int_val, 4);
            drop(Box::from_raw(a));
            drop(Box::from_raw(b));
            drop(Box::from_raw(c));
        }
    }

    #[test]
    fn test_flat() {
        // [1, [2, 3], [4, [5]]]
        let inner1 = tok_array_alloc();
        tok_array_push(inner1, TokValue::from_int(2));
        tok_array_push(inner1, TokValue::from_int(3));

        let inner2_inner = tok_array_alloc();
        tok_array_push(inner2_inner, TokValue::from_int(5));
        let inner2 = tok_array_alloc();
        tok_array_push(inner2, TokValue::from_int(4));
        tok_array_push(inner2, TokValue::from_array(inner2_inner));

        let arr = tok_array_alloc();
        tok_array_push(arr, TokValue::from_int(1));
        tok_array_push(arr, TokValue::from_array(inner1));
        tok_array_push(arr, TokValue::from_array(inner2));

        let flat = tok_array_flat(arr);
        assert_eq!(tok_array_len(flat), 5);
        unsafe {
            assert_eq!((*flat).data[0].data.int_val, 1);
            assert_eq!((*flat).data[1].data.int_val, 2);
            assert_eq!((*flat).data[2].data.int_val, 3);
            assert_eq!((*flat).data[3].data.int_val, 4);
            assert_eq!((*flat).data[4].data.int_val, 5);
            // Cleanup (simplified — skip rc for test)
            drop(Box::from_raw(flat));
            drop(Box::from_raw(arr));
            drop(Box::from_raw(inner1));
            drop(Box::from_raw(inner2));
            drop(Box::from_raw(inner2_inner));
        }
    }

    #[test]
    fn test_uniq() {
        let arr = tok_array_alloc();
        tok_array_push(arr, TokValue::from_int(1));
        tok_array_push(arr, TokValue::from_int(2));
        tok_array_push(arr, TokValue::from_int(1));
        tok_array_push(arr, TokValue::from_int(3));
        tok_array_push(arr, TokValue::from_int(2));
        let unique = tok_array_uniq(arr);
        assert_eq!(tok_array_len(unique), 3);
        unsafe {
            assert_eq!((*unique).data[0].data.int_val, 1);
            assert_eq!((*unique).data[1].data.int_val, 2);
            assert_eq!((*unique).data[2].data.int_val, 3);
            drop(Box::from_raw(arr));
            drop(Box::from_raw(unique));
        }
    }

    #[test]
    fn test_uniq_mixed_types() {
        use crate::string::TokString;
        let arr = tok_array_alloc();
        let s1 = TokValue::from_string(TokString::alloc("hello".to_string()));
        let s2 = TokValue::from_string(TokString::alloc("hello".to_string()));
        let s3 = TokValue::from_string(TokString::alloc("world".to_string()));
        tok_array_push(arr, TokValue::from_int(1));
        tok_array_push(arr, s1);
        tok_array_push(arr, TokValue::from_float(3.14));
        tok_array_push(arr, TokValue::from_int(1)); // dup int
        tok_array_push(arr, s2); // dup string by content
        tok_array_push(arr, TokValue::from_float(3.14)); // dup float
        tok_array_push(arr, s3); // different string
        tok_array_push(arr, TokValue::nil()); // nil
        tok_array_push(arr, TokValue::nil()); // dup nil
        tok_array_push(arr, TokValue::from_bool(true));
        tok_array_push(arr, TokValue::from_bool(true)); // dup bool
        let unique = tok_array_uniq(arr);
        // Expect: 1, "hello", 3.14, "world", nil, true = 6 unique
        assert_eq!(tok_array_len(unique), 6);
        unsafe {
            assert_eq!((*unique).data[0].tag, crate::value::TAG_INT);
            assert_eq!((*unique).data[1].tag, crate::value::TAG_STRING);
            assert_eq!((*unique).data[2].tag, crate::value::TAG_FLOAT);
            assert_eq!((*unique).data[3].tag, crate::value::TAG_STRING);
            assert_eq!((*unique).data[4].tag, crate::value::TAG_NIL);
            assert_eq!((*unique).data[5].tag, 3); // BOOL
            drop(Box::from_raw(arr));
            drop(Box::from_raw(unique));
        }
    }

    #[test]
    fn test_uniq_large_input() {
        let arr = tok_array_alloc();
        // 10000 elements but only 100 unique values — O(n^2) would be slow
        for i in 0..10000 {
            tok_array_push(arr, TokValue::from_int(i % 100));
        }
        let unique = tok_array_uniq(arr);
        assert_eq!(tok_array_len(unique), 100);
        unsafe {
            // Verify order preserved: first occurrence of each value
            for i in 0..100 {
                assert_eq!((*unique).data[i as usize].data.int_val, i);
            }
            drop(Box::from_raw(arr));
            drop(Box::from_raw(unique));
        }
    }

    #[test]
    fn test_join() {
        let arr = tok_array_alloc();
        let s1 = TokString::alloc("hello".to_string());
        let s2 = TokString::alloc("world".to_string());
        tok_array_push(arr, TokValue::from_string(s1));
        tok_array_push(arr, TokValue::from_string(s2));
        let sep = TokString::alloc(" ".to_string());
        let result = tok_array_join(arr, sep);
        unsafe {
            assert_eq!(&(*result).data, "hello world");
            drop(Box::from_raw(s1));
            drop(Box::from_raw(s2));
            drop(Box::from_raw(arr));
            drop(Box::from_raw(sep));
            drop(Box::from_raw(result));
        }
    }

    #[test]
    fn test_min_max_sum() {
        let arr = tok_array_alloc();
        tok_array_push(arr, TokValue::from_int(3));
        tok_array_push(arr, TokValue::from_int(1));
        tok_array_push(arr, TokValue::from_int(2));

        let min = tok_array_min(arr);
        let max = tok_array_max(arr);
        let sum = tok_array_sum(arr);
        unsafe {
            assert_eq!(min.data.int_val, 1);
            assert_eq!(max.data.int_val, 3);
            assert_eq!(sum.data.int_val, 6);
            drop(Box::from_raw(arr));
        }
    }

    #[test]
    fn test_sum_float() {
        let arr = tok_array_alloc();
        tok_array_push(arr, TokValue::from_int(1));
        tok_array_push(arr, TokValue::from_float(2.5));
        let sum = tok_array_sum(arr);
        assert_eq!(sum.tag, TAG_FLOAT);
        unsafe {
            assert!((sum.data.float_val - 3.5).abs() < 1e-10);
            drop(Box::from_raw(arr));
        }
    }

    #[test]
    fn test_empty_min_max_sum() {
        let arr = tok_array_alloc();
        let min = tok_array_min(arr);
        let max = tok_array_max(arr);
        let sum = tok_array_sum(arr);
        assert_eq!(min.tag, 0); // NIL
        assert_eq!(max.tag, 0); // NIL
        unsafe {
            assert_eq!(sum.data.int_val, 0);
        }
        unsafe {
            drop(Box::from_raw(arr));
        }
    }
}
