//! Ordered map (string keys → TokValue values) for the Tok runtime.

use std::sync::atomic::{AtomicU32, Ordering};
use indexmap::IndexMap;

use crate::value::TokValue;
use crate::string::TokString;
use crate::array::TokArray;

// ═══════════════════════════════════════════════════════════════
// TokMap
// ═══════════════════════════════════════════════════════════════

#[repr(C)]
pub struct TokMap {
    pub rc: AtomicU32,
    pub data: IndexMap<String, TokValue>,
}

impl TokMap {
    pub fn new() -> Self {
        TokMap {
            rc: AtomicU32::new(1),
            data: IndexMap::new(),
        }
    }

    pub fn rc_inc(&self) {
        self.rc.fetch_add(1, Ordering::Relaxed);
    }

    pub fn rc_dec(&self) -> bool {
        self.rc.fetch_sub(1, Ordering::Release) == 1
    }

    pub fn alloc() -> *mut TokMap {
        Box::into_raw(Box::new(TokMap::new()))
    }
}

// ═══════════════════════════════════════════════════════════════
// extern "C" API
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_map_alloc() -> *mut TokMap {
    TokMap::alloc()
}

#[no_mangle]
pub extern "C" fn tok_map_get(m: *mut TokMap, key: *mut TokString) -> TokValue {
    assert!(!m.is_null(), "tok_map_get: null map");
    assert!(!key.is_null(), "tok_map_get: null key");
    unsafe {
        let key_str = &(*key).data;
        if let Some(v) = (*m).data.get(key_str) {
            v.rc_inc();
            *v
        } else {
            TokValue::nil()
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_map_set(m: *mut TokMap, key: *mut TokString, val: TokValue) {
    assert!(!m.is_null(), "tok_map_set: null map");
    assert!(!key.is_null(), "tok_map_set: null key");
    unsafe {
        let key_str = (*key).data.clone();
        val.rc_inc();
        if let Some(old) = (*m).data.insert(key_str, val) {
            old.rc_dec();
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_map_has(m: *mut TokMap, key: *mut TokString) -> i8 {
    assert!(!m.is_null(), "tok_map_has: null map");
    assert!(!key.is_null(), "tok_map_has: null key");
    unsafe {
        let key_str = &(*key).data;
        if (*m).data.contains_key(key_str) { 1 } else { 0 }
    }
}

#[no_mangle]
pub extern "C" fn tok_map_del(m: *mut TokMap, key: *mut TokString) -> *mut TokMap {
    assert!(!m.is_null(), "tok_map_del: null map");
    assert!(!key.is_null(), "tok_map_del: null key");
    unsafe {
        let key_str = &(*key).data;
        let result = TokMap::alloc();
        for (k, v) in &(*m).data {
            if k != key_str {
                v.rc_inc();
                (*result).data.insert(k.clone(), *v);
            }
        }
        result
    }
}

#[no_mangle]
pub extern "C" fn tok_map_keys(m: *mut TokMap) -> *mut TokArray {
    assert!(!m.is_null(), "tok_map_keys: null map");
    unsafe {
        let arr = TokArray::alloc();
        for (k, _) in &(*m).data {
            let str_ptr = TokString::alloc(k.clone());
            (*arr).data.push(TokValue::from_string(str_ptr));
        }
        arr
    }
}

#[no_mangle]
pub extern "C" fn tok_map_vals(m: *mut TokMap) -> *mut TokArray {
    assert!(!m.is_null(), "tok_map_vals: null map");
    unsafe {
        let arr = TokArray::alloc();
        for (_, v) in &(*m).data {
            v.rc_inc();
            (*arr).data.push(*v);
        }
        arr
    }
}

#[no_mangle]
pub extern "C" fn tok_map_len(m: *mut TokMap) -> i64 {
    assert!(!m.is_null(), "tok_map_len: null map");
    unsafe { (*m).data.len() as i64 }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn key(s: &str) -> *mut TokString {
        TokString::alloc(s.to_string())
    }

    #[test]
    fn test_alloc_empty() {
        let m = tok_map_alloc();
        assert_eq!(tok_map_len(m), 0);
        unsafe { drop(Box::from_raw(m)); }
    }

    #[test]
    fn test_set_and_get() {
        let m = tok_map_alloc();
        let k = key("name");
        let s = TokString::alloc("Alice".to_string());
        tok_map_set(m, k, TokValue::from_string(s));
        assert_eq!(tok_map_len(m), 1);

        let v = tok_map_get(m, k);
        unsafe {
            assert_eq!(&(*v.data.string_ptr).data, "Alice");
        }

        // Overwrite
        let s2 = TokString::alloc("Bob".to_string());
        tok_map_set(m, k, TokValue::from_string(s2));
        assert_eq!(tok_map_len(m), 1);

        let v2 = tok_map_get(m, k);
        unsafe {
            assert_eq!(&(*v2.data.string_ptr).data, "Bob");
            drop(Box::from_raw(s));
            drop(Box::from_raw(s2));
            drop(Box::from_raw(k));
            drop(Box::from_raw(m));
        }
    }

    #[test]
    fn test_has() {
        let m = tok_map_alloc();
        let k = key("x");
        tok_map_set(m, k, TokValue::from_int(42));
        assert_eq!(tok_map_has(m, k), 1);

        let k2 = key("y");
        assert_eq!(tok_map_has(m, k2), 0);
        unsafe {
            drop(Box::from_raw(k));
            drop(Box::from_raw(k2));
            drop(Box::from_raw(m));
        }
    }

    #[test]
    fn test_del() {
        let m = tok_map_alloc();
        let k1 = key("a");
        let k2 = key("b");
        tok_map_set(m, k1, TokValue::from_int(1));
        tok_map_set(m, k2, TokValue::from_int(2));
        assert_eq!(tok_map_len(m), 2);

        let m2 = tok_map_del(m, k1);
        assert_eq!(tok_map_len(m2), 1);
        assert_eq!(tok_map_has(m2, k1), 0);
        assert_eq!(tok_map_has(m2, k2), 1);

        unsafe {
            drop(Box::from_raw(k1));
            drop(Box::from_raw(k2));
            drop(Box::from_raw(m));
            drop(Box::from_raw(m2));
        }
    }

    #[test]
    fn test_keys_and_vals() {
        let m = tok_map_alloc();
        let k1 = key("x");
        let k2 = key("y");
        tok_map_set(m, k1, TokValue::from_int(10));
        tok_map_set(m, k2, TokValue::from_int(20));

        let keys = tok_map_keys(m);
        let vals = tok_map_vals(m);
        unsafe {
            assert_eq!((*keys).data.len(), 2);
            assert_eq!((*vals).data.len(), 2);
            assert_eq!(&(*(*keys).data[0].data.string_ptr).data, "x");
            assert_eq!(&(*(*keys).data[1].data.string_ptr).data, "y");
            assert_eq!((*vals).data[0].data.int_val, 10);
            assert_eq!((*vals).data[1].data.int_val, 20);

            // Cleanup
            for v in &(*keys).data {
                drop(Box::from_raw(v.data.string_ptr));
            }
            drop(Box::from_raw(keys));
            drop(Box::from_raw(vals));
            drop(Box::from_raw(k1));
            drop(Box::from_raw(k2));
            drop(Box::from_raw(m));
        }
    }

    #[test]
    fn test_missing_key_returns_nil() {
        let m = tok_map_alloc();
        let k = key("nonexistent");
        let v = tok_map_get(m, k);
        assert_eq!(v.tag, 0); // NIL
        unsafe {
            drop(Box::from_raw(k));
            drop(Box::from_raw(m));
        }
    }
}
