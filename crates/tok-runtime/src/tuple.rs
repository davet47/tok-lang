//! Reference-counted tuple type for the Tok runtime.

use std::sync::atomic::{fence, AtomicU32, Ordering};

use crate::value::TokValue;

// ═══════════════════════════════════════════════════════════════
// TokTuple
// ═══════════════════════════════════════════════════════════════

#[repr(C)]
pub struct TokTuple {
    pub rc: AtomicU32,
    pub data: Vec<TokValue>,
}

impl TokTuple {
    pub fn new(elements: Vec<TokValue>) -> Self {
        TokTuple {
            rc: AtomicU32::new(1),
            data: elements,
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

    pub fn alloc(elements: Vec<TokValue>) -> *mut TokTuple {
        Box::into_raw(Box::new(TokTuple::new(elements)))
    }
}

// ═══════════════════════════════════════════════════════════════
// extern "C" API
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_tuple_alloc(count: i64) -> *mut TokTuple {
    let elems = vec![TokValue::nil(); count.max(0) as usize];
    TokTuple::alloc(elems)
}

#[no_mangle]
pub extern "C" fn tok_tuple_get(t: *mut TokTuple, idx: i64) -> TokValue {
    assert!(!t.is_null(), "tok_tuple_get: null tuple");
    unsafe {
        let i = idx as usize;
        if i < (*t).data.len() {
            let val = (*t).data[i];
            val.rc_inc();
            val
        } else {
            TokValue::nil()
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_tuple_set(t: *mut TokTuple, idx: i64, val: TokValue) {
    assert!(!t.is_null(), "tok_tuple_set: null tuple");
    unsafe {
        let i = idx as usize;
        if i < (*t).data.len() {
            let old = (*t).data[i];
            old.rc_dec();
            val.rc_inc();
            (*t).data[i] = val;
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_tuple_len(t: *mut TokTuple) -> i64 {
    assert!(!t.is_null(), "tok_tuple_len: null tuple");
    unsafe { (*t).data.len() as i64 }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_access() {
        let t = tok_tuple_alloc(3);
        tok_tuple_set(t, 0, TokValue::from_int(10));
        tok_tuple_set(t, 1, TokValue::from_int(20));
        tok_tuple_set(t, 2, TokValue::from_int(30));
        assert_eq!(tok_tuple_len(t), 3);

        let v0 = tok_tuple_get(t, 0);
        let v1 = tok_tuple_get(t, 1);
        unsafe {
            assert_eq!(v0.data.int_val, 10);
            assert_eq!(v1.data.int_val, 20);
            drop(Box::from_raw(t));
        }
    }

    #[test]
    fn test_out_of_bounds() {
        let t = tok_tuple_alloc(1);
        tok_tuple_set(t, 0, TokValue::from_int(42));
        let v = tok_tuple_get(t, 5);
        assert_eq!(v.tag, 0); // NIL
        unsafe {
            drop(Box::from_raw(t));
        }
    }
}
