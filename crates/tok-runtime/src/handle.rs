//! Handle type for goroutine results in the Tok runtime.

use std::sync::atomic::{fence, AtomicU32, Ordering};
use std::sync::Mutex;
use std::thread::JoinHandle;

use crate::value::TokValue;

// ═══════════════════════════════════════════════════════════════
// TokHandle
// ═══════════════════════════════════════════════════════════════

pub struct TokHandle {
    pub rc: AtomicU32,
    pub handle: Mutex<Option<JoinHandle<TokValue>>>,
}

// Safety: JoinHandle is Send, and we protect it with Mutex.
unsafe impl Send for TokHandle {}
unsafe impl Sync for TokHandle {}

impl TokHandle {
    pub fn new(handle: JoinHandle<TokValue>) -> Self {
        TokHandle {
            rc: AtomicU32::new(1),
            handle: Mutex::new(Some(handle)),
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

    pub fn alloc(handle: JoinHandle<TokValue>) -> *mut TokHandle {
        Box::into_raw(Box::new(TokHandle::new(handle)))
    }

    /// Join the thread, returning its result. Can only be called once.
    pub fn join(&self) -> TokValue {
        let mut guard = self.handle.lock().unwrap_or_else(|e| e.into_inner());
        match guard.take() {
            Some(h) => h.join().unwrap_or_default(),
            None => TokValue::nil(), // Already joined
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// extern "C" API
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_go(
    fn_ptr: extern "C" fn(*mut u8) -> TokValue,
    env: *mut u8,
) -> *mut TokHandle {
    // Safety: env is passed to the spawned thread. The caller is responsible
    // for ensuring the env allocation lives long enough.
    let env_usize = env as usize;
    let handle = std::thread::spawn(move || {
        let env_ptr = env_usize as *mut u8;
        fn_ptr(env_ptr)
    });
    TokHandle::alloc(handle)
}

#[no_mangle]
pub extern "C" fn tok_handle_join(h: *mut TokHandle) -> TokValue {
    null_check!(h, "tok_handle_join: null handle");
    unsafe { (*h).join() }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_go_and_join() {
        extern "C" fn add_one(env: *mut u8) -> TokValue {
            let val = env as i64;
            TokValue::from_int(val + 1)
        }

        let h = tok_go(add_one, 41 as *mut u8);
        let result = tok_handle_join(h);
        unsafe {
            assert_eq!(result.data.int_val, 42);
            drop(Box::from_raw(h));
        }
    }

    #[test]
    fn test_double_join_returns_nil() {
        extern "C" fn ret_ten(_env: *mut u8) -> TokValue {
            TokValue::from_int(10)
        }

        let h = tok_go(ret_ten, std::ptr::null_mut());
        let v1 = tok_handle_join(h);
        unsafe {
            assert_eq!(v1.data.int_val, 10);
        }

        // Second join returns nil
        let v2 = tok_handle_join(h);
        assert_eq!(v2.tag, 0); // NIL

        unsafe {
            drop(Box::from_raw(h));
        }
    }
}
