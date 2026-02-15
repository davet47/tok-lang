//! Channel type for the Tok runtime.
//!
//! Supports both unbuffered (capacity=0, rendezvous) and
//! buffered (capacity>0, VecDeque) semantics.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};

use crate::value::TokValue;

// ═══════════════════════════════════════════════════════════════
// TokChannel
// ═══════════════════════════════════════════════════════════════

struct ChannelInner {
    buffer: VecDeque<TokValue>,
    capacity: usize,
    /// For unbuffered channels: a sender is waiting for a receiver.
    sender_waiting: Option<TokValue>,
    /// For unbuffered channels: signals the sender picked up.
    sender_ready: bool,
    /// Reserved for future channel close semantics.
    #[allow(dead_code)]
    closed: bool,
}

pub struct TokChannel {
    pub rc: AtomicU32,
    inner: Mutex<ChannelInner>,
    /// Notifies receivers that data is available.
    recv_condvar: Condvar,
    /// Notifies senders that space is available (or receiver picked up).
    send_condvar: Condvar,
}

// Safety: TokChannel uses Mutex+Condvar for synchronization.
unsafe impl Send for TokChannel {}
unsafe impl Sync for TokChannel {}

impl TokChannel {
    pub fn new(capacity: usize) -> Self {
        TokChannel {
            rc: AtomicU32::new(1),
            inner: Mutex::new(ChannelInner {
                buffer: VecDeque::new(),
                capacity,
                sender_waiting: None,
                sender_ready: false,
                closed: false,
            }),
            recv_condvar: Condvar::new(),
            send_condvar: Condvar::new(),
        }
    }

    pub fn rc_inc(&self) {
        self.rc.fetch_add(1, Ordering::Relaxed);
    }

    pub fn rc_dec(&self) -> bool {
        self.rc.fetch_sub(1, Ordering::Release) == 1
    }

    pub fn alloc(capacity: usize) -> *mut TokChannel {
        Box::into_raw(Box::new(TokChannel::new(capacity)))
    }

    /// Blocking send.
    pub fn send(&self, val: TokValue) {
        let mut inner = self.inner.lock().unwrap();
        if inner.capacity == 0 {
            // Unbuffered: rendezvous
            // Wait until no other sender is waiting
            while inner.sender_waiting.is_some() {
                inner = self.send_condvar.wait(inner).unwrap();
            }
            inner.sender_waiting = Some(val);
            inner.sender_ready = false;
            self.recv_condvar.notify_one();
            // Wait until receiver picks it up
            while !inner.sender_ready {
                inner = self.send_condvar.wait(inner).unwrap();
            }
            inner.sender_ready = false;
        } else {
            // Buffered: wait until space available
            while inner.buffer.len() >= inner.capacity {
                inner = self.send_condvar.wait(inner).unwrap();
            }
            inner.buffer.push_back(val);
            self.recv_condvar.notify_one();
        }
    }

    /// Blocking receive.
    pub fn recv(&self) -> TokValue {
        let mut inner = self.inner.lock().unwrap();
        if inner.capacity == 0 {
            // Unbuffered: rendezvous
            while inner.sender_waiting.is_none() {
                inner = self.recv_condvar.wait(inner).unwrap();
            }
            let val = inner.sender_waiting.take().unwrap();
            inner.sender_ready = true;
            self.send_condvar.notify_one();
            val
        } else {
            // Buffered: wait until data available
            while inner.buffer.is_empty() {
                inner = self.recv_condvar.wait(inner).unwrap();
            }
            let val = inner.buffer.pop_front().unwrap();
            self.send_condvar.notify_one();
            val
        }
    }

    /// Non-blocking try_send. Returns true if sent.
    pub fn try_send(&self, val: TokValue) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.capacity == 0 {
            // Unbuffered: can only send if a receiver is already waiting
            // This is a simplification — in practice we check if someone
            // is blocked on recv. For select semantics, we just return false
            // if no receiver is waiting.
            false
        } else {
            if inner.buffer.len() < inner.capacity {
                inner.buffer.push_back(val);
                self.recv_condvar.notify_one();
                true
            } else {
                false
            }
        }
    }

    /// Non-blocking try_recv. Returns true if received.
    pub fn try_recv(&self) -> Option<TokValue> {
        let mut inner = self.inner.lock().unwrap();
        if inner.capacity == 0 {
            // Unbuffered: pick up a waiting sender's value
            if inner.sender_waiting.is_some() {
                let val = inner.sender_waiting.take().unwrap();
                inner.sender_ready = true;
                self.send_condvar.notify_one();
                Some(val)
            } else {
                None
            }
        } else {
            if let Some(val) = inner.buffer.pop_front() {
                self.send_condvar.notify_one();
                Some(val)
            } else {
                None
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// extern "C" API
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_channel_alloc(capacity: i64) -> *mut TokChannel {
    let cap = if capacity < 0 { 0 } else { capacity as usize };
    TokChannel::alloc(cap)
}

#[no_mangle]
pub extern "C" fn tok_channel_send(ch: *mut TokChannel, val: TokValue) {
    assert!(!ch.is_null(), "tok_channel_send: null channel");
    unsafe { (*ch).send(val); }
}

#[no_mangle]
pub extern "C" fn tok_channel_recv(ch: *mut TokChannel) -> TokValue {
    assert!(!ch.is_null(), "tok_channel_recv: null channel");
    unsafe { (*ch).recv() }
}

#[no_mangle]
pub extern "C" fn tok_channel_try_send(ch: *mut TokChannel, val: TokValue) -> i8 {
    assert!(!ch.is_null(), "tok_channel_try_send: null channel");
    unsafe {
        if (*ch).try_send(val) {
            1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_channel_try_recv(ch: *mut TokChannel, out: *mut TokValue) -> i8 {
    assert!(!ch.is_null(), "tok_channel_try_recv: null channel");
    unsafe {
        match (*ch).try_recv() {
            Some(val) => {
                if !out.is_null() {
                    *out = val;
                }
                1
            }
            None => 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_buffered_send_recv() {
        let ch = tok_channel_alloc(2);
        tok_channel_send(ch, TokValue::from_int(10));
        tok_channel_send(ch, TokValue::from_int(20));
        let v1 = tok_channel_recv(ch);
        let v2 = tok_channel_recv(ch);
        unsafe {
            assert_eq!(v1.data.int_val, 10);
            assert_eq!(v2.data.int_val, 20);
            drop(Box::from_raw(ch));
        }
    }

    #[test]
    fn test_buffered_try_send_recv() {
        let ch = tok_channel_alloc(1);
        assert_eq!(tok_channel_try_send(ch, TokValue::from_int(42)), 1);
        // Buffer full
        assert_eq!(tok_channel_try_send(ch, TokValue::from_int(99)), 0);

        let mut out = TokValue::nil();
        assert_eq!(tok_channel_try_recv(ch, &mut out), 1);
        unsafe { assert_eq!(out.data.int_val, 42); }

        // Empty
        assert_eq!(tok_channel_try_recv(ch, &mut out), 0);
        unsafe { drop(Box::from_raw(ch)); }
    }

    #[test]
    fn test_unbuffered_send_recv() {
        let ch = tok_channel_alloc(0);
        let ch_ptr = ch as usize;

        let sender = thread::spawn(move || {
            let ch = ch_ptr as *mut TokChannel;
            tok_channel_send(ch, TokValue::from_int(42));
        });

        // Give sender a moment to start
        thread::sleep(std::time::Duration::from_millis(10));

        let val = tok_channel_recv(ch);
        unsafe { assert_eq!(val.data.int_val, 42); }

        sender.join().unwrap();
        unsafe { drop(Box::from_raw(ch)); }
    }

    #[test]
    fn test_unbuffered_try_recv_empty() {
        let ch = tok_channel_alloc(0);
        let mut out = TokValue::nil();
        assert_eq!(tok_channel_try_recv(ch, &mut out), 0);
        unsafe { drop(Box::from_raw(ch)); }
    }

    #[test]
    fn test_multiple_buffered() {
        let ch = tok_channel_alloc(10);
        for i in 0..10 {
            tok_channel_send(ch, TokValue::from_int(i));
        }
        for i in 0..10 {
            let v = tok_channel_recv(ch);
            unsafe { assert_eq!(v.data.int_val, i); }
        }
        unsafe { drop(Box::from_raw(ch)); }
    }
}
