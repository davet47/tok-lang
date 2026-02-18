//! Reference-counted string type for the Tok runtime.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::array::TokArray;
use crate::value::TokValue;

// ═══════════════════════════════════════════════════════════════
// TokString
// ═══════════════════════════════════════════════════════════════

#[repr(C)]
pub struct TokString {
    pub rc: AtomicU32,
    pub data: String,
}

impl TokString {
    pub fn new(s: String) -> Self {
        TokString {
            rc: AtomicU32::new(1),
            data: s,
        }
    }

    pub fn rc_inc(&self) {
        self.rc.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement refcount. Returns true if the object should be freed.
    pub fn rc_dec(&self) -> bool {
        self.rc.fetch_sub(1, Ordering::Release) == 1
    }

    pub fn alloc(s: String) -> *mut TokString {
        Box::into_raw(Box::new(TokString::new(s)))
    }
}

// ═══════════════════════════════════════════════════════════════
// extern "C" API
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_string_alloc(data: *const u8, len: usize) -> *mut TokString {
    let s = if data.is_null() || len == 0 {
        String::new()
    } else {
        unsafe {
            let slice = std::slice::from_raw_parts(data, len);
            String::from_utf8_lossy(slice).into_owned()
        }
    };
    TokString::alloc(s)
}

#[no_mangle]
pub extern "C" fn tok_string_concat(a: *mut TokString, b: *mut TokString) -> *mut TokString {
    assert!(!a.is_null(), "tok_string_concat: null lhs");
    assert!(!b.is_null(), "tok_string_concat: null rhs");
    unsafe {
        // COW optimization: if `a` has refcount 1, nobody else holds a reference,
        // so we can mutate in-place and return the same pointer. This turns
        // O(n²) repeated concat into amortized O(n) via String's growth strategy.
        if (*a).rc.load(Ordering::Relaxed) == 1 {
            (*a).data.push_str(&(*b).data);
            a
        } else {
            let mut result = String::with_capacity((*a).data.len() + (*b).data.len());
            result.push_str(&(*a).data);
            result.push_str(&(*b).data);
            TokString::alloc(result)
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_string_len(s: *mut TokString) -> i64 {
    assert!(!s.is_null(), "tok_string_len: null pointer");
    unsafe { (*s).data.chars().count() as i64 }
}

#[no_mangle]
pub extern "C" fn tok_string_eq(a: *mut TokString, b: *mut TokString) -> i8 {
    if a.is_null() && b.is_null() {
        return 1;
    }
    if a.is_null() || b.is_null() {
        return 0;
    }
    unsafe {
        if (*a).data == (*b).data {
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_string_cmp(a: *mut TokString, b: *mut TokString) -> i64 {
    assert!(!a.is_null(), "tok_string_cmp: null lhs");
    assert!(!b.is_null(), "tok_string_cmp: null rhs");
    unsafe {
        match (*a).data.cmp(&(*b).data) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_string_index(s: *mut TokString, i: i64) -> *mut TokString {
    assert!(!s.is_null(), "tok_string_index: null pointer");
    unsafe {
        let chars: Vec<char> = (*s).data.chars().collect();
        let idx = if i < 0 {
            (chars.len() as i64 + i) as usize
        } else {
            i as usize
        };
        if idx < chars.len() {
            TokString::alloc(chars[idx].to_string())
        } else {
            TokString::alloc(String::new())
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_string_repeat(s: *mut TokString, count: i64) -> *mut TokString {
    assert!(!s.is_null(), "tok_string_repeat: null pointer");
    unsafe {
        let n = count.max(0) as usize;
        TokString::alloc((*s).data.repeat(n))
    }
}

#[no_mangle]
pub extern "C" fn tok_string_slice(s: *mut TokString, start: i64, end: i64) -> *mut TokString {
    assert!(!s.is_null(), "tok_string_slice: null pointer");
    unsafe {
        let chars: Vec<char> = (*s).data.chars().collect();
        let len = chars.len() as i64;
        let s_idx = start.max(0).min(len) as usize;
        let e_idx = end.max(0).min(len) as usize;
        if s_idx >= e_idx {
            TokString::alloc(String::new())
        } else {
            let result: String = chars[s_idx..e_idx].iter().collect();
            TokString::alloc(result)
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_string_split(s: *mut TokString, delim: *mut TokString) -> *mut TokArray {
    assert!(!s.is_null(), "tok_string_split: null string");
    assert!(!delim.is_null(), "tok_string_split: null delimiter");
    unsafe {
        let src = &(*s).data;
        let delim_str = &(*delim).data;

        // Pre-count splits to allocate array with exact capacity
        let count = src.matches(delim_str).count() + 1;
        let mut data = Vec::with_capacity(count);

        // Iterate directly without intermediate Vec<&str>
        for part in src.split(delim_str) {
            let str_ptr = TokString::alloc(part.to_string());
            data.push(TokValue::from_string(str_ptr));
        }

        Box::into_raw(Box::new(TokArray {
            rc: std::sync::atomic::AtomicU32::new(1),
            data,
        }))
    }
}

#[no_mangle]
pub extern "C" fn tok_string_trim(s: *mut TokString) -> *mut TokString {
    assert!(!s.is_null(), "tok_string_trim: null pointer");
    unsafe {
        let trimmed = (*s).data.trim().to_string();
        TokString::alloc(trimmed)
    }
}

#[no_mangle]
pub extern "C" fn tok_string_get_ptr(s: *mut TokString) -> *const u8 {
    assert!(!s.is_null(), "tok_string_get_ptr: null pointer");
    unsafe { (*s).data.as_ptr() }
}

#[no_mangle]
pub extern "C" fn tok_string_get_len(s: *mut TokString) -> usize {
    assert!(!s.is_null(), "tok_string_get_len: null pointer");
    unsafe { (*s).data.len() }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn alloc_str(s: &str) -> *mut TokString {
        TokString::alloc(s.to_string())
    }

    /// Helper to free a TokString (for test cleanup).
    unsafe fn free_str(p: *mut TokString) {
        drop(Box::from_raw(p));
    }

    #[test]
    fn test_alloc_and_content() {
        let s = alloc_str("hello");
        unsafe {
            assert_eq!(&(*s).data, "hello");
            assert_eq!((*s).rc.load(Ordering::Relaxed), 1);
            free_str(s);
        }
    }

    #[test]
    fn test_concat() {
        let a = alloc_str("hello ");
        let b = alloc_str("world");
        let c = tok_string_concat(a, b);
        unsafe {
            assert_eq!(&(*c).data, "hello world");
            // COW may reuse `a`, so c may equal a
            if c != a {
                free_str(a);
            }
            free_str(b);
            free_str(c);
        }
    }

    #[test]
    fn test_len() {
        let s = alloc_str("hello");
        assert_eq!(tok_string_len(s), 5);
        unsafe {
            free_str(s);
        }
    }

    #[test]
    fn test_len_unicode() {
        let s = alloc_str("cafe\u{0301}"); // "cafe" + combining accent
                                           // char count = 5 (c, a, f, e, combining_accent)
        assert_eq!(tok_string_len(s), 5);
        unsafe {
            free_str(s);
        }
    }

    #[test]
    fn test_eq() {
        let a = alloc_str("hello");
        let b = alloc_str("hello");
        let c = alloc_str("world");
        assert_eq!(tok_string_eq(a, b), 1);
        assert_eq!(tok_string_eq(a, c), 0);
        unsafe {
            free_str(a);
            free_str(b);
            free_str(c);
        }
    }

    #[test]
    fn test_slice() {
        let s = alloc_str("hello world");
        let sliced = tok_string_slice(s, 0, 5);
        unsafe {
            assert_eq!(&(*sliced).data, "hello");
            free_str(s);
            free_str(sliced);
        }
    }

    #[test]
    fn test_split() {
        let s = alloc_str("a,b,c");
        let d = alloc_str(",");
        let arr = tok_string_split(s, d);
        unsafe {
            assert_eq!((*arr).data.len(), 3);
            assert_eq!(&(*(*arr).data[0].data.string_ptr).data, "a");
            assert_eq!(&(*(*arr).data[1].data.string_ptr).data, "b");
            assert_eq!(&(*(*arr).data[2].data.string_ptr).data, "c");
            // Cleanup
            for v in &(*arr).data {
                drop(Box::from_raw(v.data.string_ptr));
            }
            drop(Box::from_raw(arr));
            free_str(s);
            free_str(d);
        }
    }

    #[test]
    fn test_trim() {
        let s = alloc_str("  hello  ");
        let trimmed = tok_string_trim(s);
        unsafe {
            assert_eq!(&(*trimmed).data, "hello");
            free_str(s);
            free_str(trimmed);
        }
    }

    #[test]
    fn test_index() {
        let s = alloc_str("hello");
        let ch = tok_string_index(s, 1);
        unsafe {
            assert_eq!(&(*ch).data, "e");
            free_str(s);
            free_str(ch);
        }
    }

    #[test]
    fn test_cmp() {
        let a = alloc_str("abc");
        let b = alloc_str("def");
        assert_eq!(tok_string_cmp(a, b), -1);
        assert_eq!(tok_string_cmp(b, a), 1);
        assert_eq!(tok_string_cmp(a, a), 0);
        unsafe {
            free_str(a);
            free_str(b);
        }
    }

    #[test]
    fn test_refcount() {
        let s = alloc_str("test");
        unsafe {
            assert_eq!((*s).rc.load(Ordering::Relaxed), 1);
            (*s).rc_inc();
            assert_eq!((*s).rc.load(Ordering::Relaxed), 2);
            assert!(!(*s).rc_dec()); // 2 -> 1, not freed
            assert!((*s).rc_dec()); // 1 -> 0, should free
                                    // Actually free it
            drop(Box::from_raw(s));
        }
    }
}
