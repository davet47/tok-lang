//! Channel type for the Tok runtime.
//!
//! Supports both unbuffered (capacity=0, rendezvous) and
//! buffered (capacity>0) semantics.
//!
//! Buffered channels use a Mutex-protected VecDeque, safe for
//! multiple producers and multiple consumers (MPMC).

use std::collections::VecDeque;
use std::sync::atomic::{fence, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};

use crate::value::TokValue;

// ═══════════════════════════════════════════════════════════════
// TokChannel
// ═══════════════════════════════════════════════════════════════

/// Buffered channel state, protected by Mutex.
struct BufferedInner {
    buf: VecDeque<TokValue>,
    capacity: usize,
}

/// Unbuffered channel state (rendezvous), protected by Mutex.
struct UnbufferedInner {
    sender_waiting: Option<TokValue>,
    sender_ready: bool,
    #[allow(dead_code)]
    closed: bool,
}

pub struct TokChannel {
    pub rc: AtomicU32,
    capacity: usize,
    /// Mutex-protected bounded queue (used when capacity > 0).
    buffered: Option<Mutex<BufferedInner>>,
    /// Mutex-based state for unbuffered channels (capacity == 0).
    unbuffered: Option<Mutex<UnbufferedInner>>,
    /// Notifies receivers that data is available.
    recv_condvar: Condvar,
    /// Notifies senders that space is available (or receiver picked up).
    send_condvar: Condvar,
}

// Safety: TokChannel uses Mutex+Condvar for all synchronization.
unsafe impl Send for TokChannel {}
unsafe impl Sync for TokChannel {}

impl TokChannel {
    pub fn new(capacity: usize) -> Self {
        if capacity == 0 {
            TokChannel {
                rc: AtomicU32::new(1),
                capacity,
                buffered: None,
                unbuffered: Some(Mutex::new(UnbufferedInner {
                    sender_waiting: None,
                    sender_ready: false,
                    closed: false,
                })),
                recv_condvar: Condvar::new(),
                send_condvar: Condvar::new(),
            }
        } else {
            TokChannel {
                rc: AtomicU32::new(1),
                capacity,
                buffered: Some(Mutex::new(BufferedInner {
                    buf: VecDeque::with_capacity(capacity),
                    capacity,
                })),
                unbuffered: None,
                recv_condvar: Condvar::new(),
                send_condvar: Condvar::new(),
            }
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

    pub fn alloc(capacity: usize) -> *mut TokChannel {
        Box::into_raw(Box::new(TokChannel::new(capacity)))
    }

    /// Blocking send.
    pub fn send(&self, val: TokValue) {
        if self.capacity == 0 {
            self.send_unbuffered(val);
        } else {
            self.send_buffered(val);
        }
    }

    /// Blocking receive.
    pub fn recv(&self) -> TokValue {
        if self.capacity == 0 {
            self.recv_unbuffered()
        } else {
            self.recv_buffered()
        }
    }

    fn send_buffered(&self, val: TokValue) {
        let buf_mutex = self
            .buffered
            .as_ref()
            .expect("channel: buffered state missing on buffered channel");
        let mut inner = buf_mutex.lock().unwrap_or_else(|e| e.into_inner());
        // Wait until there's space
        while inner.buf.len() >= inner.capacity {
            inner = self
                .send_condvar
                .wait(inner)
                .unwrap_or_else(|e| e.into_inner());
        }
        inner.buf.push_back(val);
        self.recv_condvar.notify_one();
    }

    fn recv_buffered(&self) -> TokValue {
        let buf_mutex = self
            .buffered
            .as_ref()
            .expect("channel: buffered state missing on buffered channel");
        let mut inner = buf_mutex.lock().unwrap_or_else(|e| e.into_inner());
        // Wait until there's data
        while inner.buf.is_empty() {
            inner = self
                .recv_condvar
                .wait(inner)
                .unwrap_or_else(|e| e.into_inner());
        }
        let val = inner
            .buf
            .pop_front()
            .expect("channel: buffer was empty after non-empty check");
        self.send_condvar.notify_one();
        val
    }

    fn send_unbuffered(&self, val: TokValue) {
        let unbuf = self
            .unbuffered
            .as_ref()
            .expect("channel: unbuffered state missing");
        let mut inner = unbuf.lock().unwrap_or_else(|e| e.into_inner());
        while inner.sender_waiting.is_some() {
            inner = self
                .send_condvar
                .wait(inner)
                .unwrap_or_else(|e| e.into_inner());
        }
        inner.sender_waiting = Some(val);
        inner.sender_ready = false;
        self.recv_condvar.notify_one();
        while !inner.sender_ready {
            inner = self
                .send_condvar
                .wait(inner)
                .unwrap_or_else(|e| e.into_inner());
        }
        inner.sender_ready = false;
    }

    fn recv_unbuffered(&self) -> TokValue {
        let unbuf = self
            .unbuffered
            .as_ref()
            .expect("channel: unbuffered state missing");
        let mut inner = unbuf.lock().unwrap_or_else(|e| e.into_inner());
        while inner.sender_waiting.is_none() {
            inner = self
                .recv_condvar
                .wait(inner)
                .unwrap_or_else(|e| e.into_inner());
        }
        let val = inner
            .sender_waiting
            .take()
            .expect("channel: sender_waiting was None after is_some check");
        inner.sender_ready = true;
        self.send_condvar.notify_one();
        val
    }

    /// Non-blocking try_send. Returns true if sent.
    pub fn try_send(&self, val: TokValue) -> bool {
        if self.capacity == 0 {
            false // Unbuffered: can't non-blocking send
        } else {
            let buf_mutex = self
                .buffered
                .as_ref()
                .expect("channel: buffered state missing on buffered channel");
            let mut inner = buf_mutex.lock().unwrap_or_else(|e| e.into_inner());
            if inner.buf.len() < inner.capacity {
                inner.buf.push_back(val);
                self.recv_condvar.notify_one();
                true
            } else {
                false
            }
        }
    }

    /// Non-blocking try_recv.
    pub fn try_recv(&self) -> Option<TokValue> {
        if self.capacity == 0 {
            let unbuf = self
                .unbuffered
                .as_ref()
                .expect("channel: unbuffered state missing");
            let mut inner = unbuf.lock().unwrap_or_else(|e| e.into_inner());
            if inner.sender_waiting.is_some() {
                let val = inner
                    .sender_waiting
                    .take()
                    .expect("channel: sender_waiting was None after is_some check");
                inner.sender_ready = true;
                self.send_condvar.notify_one();
                Some(val)
            } else {
                None
            }
        } else {
            let buf_mutex = self
                .buffered
                .as_ref()
                .expect("channel: buffered state missing on buffered channel");
            let mut inner = buf_mutex.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(val) = inner.buf.pop_front() {
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
    unsafe {
        (*ch).send(val);
    }
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
        // Buffer is now full (capacity=1)
        assert_eq!(tok_channel_try_send(ch, TokValue::from_int(99)), 0);
        let mut out = TokValue::nil();
        assert_eq!(tok_channel_try_recv(ch, &mut out), 1);
        unsafe {
            assert_eq!(out.data.int_val, 42);
        }

        // Empty
        assert_eq!(tok_channel_try_recv(ch, &mut out), 0);
        unsafe {
            drop(Box::from_raw(ch));
        }
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
        unsafe {
            assert_eq!(val.data.int_val, 42);
        }

        sender.join().unwrap();
        unsafe {
            drop(Box::from_raw(ch));
        }
    }

    #[test]
    fn test_unbuffered_try_recv_empty() {
        let ch = tok_channel_alloc(0);
        let mut out = TokValue::nil();
        assert_eq!(tok_channel_try_recv(ch, &mut out), 0);
        unsafe {
            drop(Box::from_raw(ch));
        }
    }

    #[test]
    fn test_multiple_buffered() {
        let ch = tok_channel_alloc(10);
        for i in 0..10 {
            tok_channel_send(ch, TokValue::from_int(i));
        }
        for i in 0..10 {
            let v = tok_channel_recv(ch);
            unsafe {
                assert_eq!(v.data.int_val, i);
            }
        }
        unsafe {
            drop(Box::from_raw(ch));
        }
    }

    #[test]
    fn test_mpmc_threaded() {
        // Test MPMC: multiple producers + multiple consumers
        let ch = tok_channel_alloc(100);
        let ch_usize = ch as usize;
        let n_producers = 4;
        let msgs_per_producer: i64 = 1000;

        let producers: Vec<_> = (0..n_producers)
            .map(|_| {
                let ch_ptr = ch_usize;
                thread::spawn(move || {
                    let ch = ch_ptr as *mut TokChannel;
                    for i in 0..msgs_per_producer {
                        tok_channel_send(ch, TokValue::from_int(i));
                    }
                })
            })
            .collect();

        let total = n_producers * msgs_per_producer;
        let mut sum: i64 = 0;
        for _ in 0..total {
            let v = tok_channel_recv(ch);
            unsafe {
                sum += v.data.int_val;
            }
        }

        for p in producers {
            p.join().unwrap();
        }
        // Each producer sends 0..1000, sum = 999*1000/2 = 499500, times 4 = 1998000
        let expected = n_producers * (msgs_per_producer - 1) * msgs_per_producer / 2;
        assert_eq!(sum, expected);
        unsafe {
            drop(Box::from_raw(ch));
        }
    }

    #[test]
    fn test_spsc_threaded() {
        let ch = tok_channel_alloc(100);
        let ch_ptr = ch as usize;
        let n = 10000;

        let sender = thread::spawn(move || {
            let ch = ch_ptr as *mut TokChannel;
            for i in 0..n {
                tok_channel_send(ch, TokValue::from_int(i));
            }
        });

        let mut sum: i64 = 0;
        for _ in 0..n {
            let v = tok_channel_recv(ch);
            unsafe {
                sum += v.data.int_val;
            }
        }

        sender.join().unwrap();
        assert_eq!(sum, (n - 1) * n / 2);
        unsafe {
            drop(Box::from_raw(ch));
        }
    }
}
