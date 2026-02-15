//! Closure type for the Tok runtime.
//!
//! A closure pairs a function pointer with a captured environment.

use std::sync::atomic::{AtomicU32, Ordering};

// ═══════════════════════════════════════════════════════════════
// TokClosure
// ═══════════════════════════════════════════════════════════════

#[repr(C)]
pub struct TokClosure {
    pub rc: AtomicU32,
    /// Pointer to the compiled function.
    pub fn_ptr: *const u8,
    /// Pointer to the captured environment (heap-allocated by codegen).
    pub env_ptr: *mut u8,
    /// Number of parameters the function accepts.
    pub arity: u32,
}

// Safety: fn_ptr is a code pointer (immutable), env_ptr is a heap
// allocation whose lifetime is managed by the refcount.
unsafe impl Send for TokClosure {}
unsafe impl Sync for TokClosure {}

impl TokClosure {
    pub fn new(fn_ptr: *const u8, env_ptr: *mut u8, arity: u32) -> Self {
        TokClosure {
            rc: AtomicU32::new(1),
            fn_ptr,
            env_ptr,
            arity,
        }
    }

    pub fn rc_inc(&self) {
        self.rc.fetch_add(1, Ordering::Relaxed);
    }

    pub fn rc_dec(&self) -> bool {
        self.rc.fetch_sub(1, Ordering::Release) == 1
    }

    pub fn alloc(fn_ptr: *const u8, env_ptr: *mut u8, arity: u32) -> *mut TokClosure {
        Box::into_raw(Box::new(TokClosure::new(fn_ptr, env_ptr, arity)))
    }
}

// ═══════════════════════════════════════════════════════════════
// extern "C" API
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_closure_alloc(
    fn_ptr: *const u8,
    env_ptr: *mut u8,
    arity: u32,
) -> *mut TokClosure {
    TokClosure::alloc(fn_ptr, env_ptr, arity)
}

#[no_mangle]
pub extern "C" fn tok_closure_get_fn(c: *mut TokClosure) -> *const u8 {
    assert!(!c.is_null(), "tok_closure_get_fn: null closure");
    unsafe { (*c).fn_ptr }
}

#[no_mangle]
pub extern "C" fn tok_closure_get_env(c: *mut TokClosure) -> *mut u8 {
    assert!(!c.is_null(), "tok_closure_get_env: null closure");
    unsafe { (*c).env_ptr }
}

/// Allocate a closure environment: a contiguous block of `count` TokValues (16 bytes each).
/// Returns a pointer to the start of the environment.
#[no_mangle]
pub extern "C" fn tok_env_alloc(count: i64) -> *mut u8 {
    if count <= 0 {
        return std::ptr::null_mut();
    }
    let size = (count as usize) * 16; // each TokValue is 16 bytes (tag: i64 + data: i64)
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        ptr
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc() {
        let c = tok_closure_alloc(0x1234 as *const u8, std::ptr::null_mut(), 2);
        unsafe {
            assert_eq!((*c).fn_ptr, 0x1234 as *const u8);
            assert_eq!((*c).env_ptr, std::ptr::null_mut());
            assert_eq!((*c).arity, 2);
            assert_eq!((*c).rc.load(Ordering::Relaxed), 1);
            drop(Box::from_raw(c));
        }
    }

    #[test]
    fn test_getters() {
        let c = tok_closure_alloc(0xABCD as *const u8, 0xEF01 as *mut u8, 3);
        assert_eq!(tok_closure_get_fn(c), 0xABCD as *const u8);
        assert_eq!(tok_closure_get_env(c), 0xEF01 as *mut u8);
        unsafe { drop(Box::from_raw(c)); }
    }
}
