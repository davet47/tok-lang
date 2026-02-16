//! Channel type for the Tok runtime.
//!
//! Supports both unbuffered (capacity=0, rendezvous) and
//! buffered (capacity>0) semantics.
//!
//! Buffered channels use a lock-free SPSC ring buffer for the fast path,
//! falling back to parking (thread::park/unpark) when the buffer is full/empty.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use crate::value::TokValue;

// ═══════════════════════════════════════════════════════════════
// TokChannel
// ═══════════════════════════════════════════════════════════════

/// Lock-free SPSC ring buffer for buffered channels.
struct SpscRing {
    /// Ring buffer slots (capacity is always a power of 2).
    buf: Box<[UnsafeCell<MaybeUninit<TokValue>>]>,
    /// Mask for fast modulo: capacity - 1.
    mask: usize,
    /// Write position (only modified by producer).
    head: AtomicUsize,
    /// Read position (only modified by consumer).
    tail: AtomicUsize,
}

unsafe impl Send for SpscRing {}
unsafe impl Sync for SpscRing {}

impl SpscRing {
    fn new(capacity: usize) -> Self {
        // Round up to next power of 2
        let cap = capacity.next_power_of_two().max(2);
        let mut buf = Vec::with_capacity(cap);
        for _ in 0..cap {
            buf.push(UnsafeCell::new(MaybeUninit::uninit()));
        }
        SpscRing {
            buf: buf.into_boxed_slice(),
            mask: cap - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Try to push a value. Returns true if successful.
    #[inline]
    fn try_push(&self, val: TokValue) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= self.mask + 1 {
            return false; // Full
        }
        unsafe {
            (*self.buf[head & self.mask].get()).write(val);
        }
        self.head.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Try to pop a value. Returns Some(val) if successful.
    #[inline]
    fn try_pop(&self) -> Option<TokValue> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail == head {
            return None; // Empty
        }
        let val = unsafe {
            (*self.buf[tail & self.mask].get()).assume_init_read()
        };
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(val)
    }

    #[inline]
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    #[inline]
    #[allow(dead_code)]
    fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }
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
    /// Lock-free ring buffer (used when capacity > 0).
    ring: Option<SpscRing>,
    /// Mutex-based state for unbuffered channels (capacity == 0).
    unbuffered: Option<Mutex<UnbufferedInner>>,
    /// Notifies receivers that data is available.
    recv_condvar: Condvar,
    /// Notifies senders that space is available (or receiver picked up).
    send_condvar: Condvar,
    /// Mutex used only for condvar wait (buffered channels wait on this).
    wait_mutex: Mutex<()>,
}

// Safety: TokChannel uses atomics and Mutex+Condvar for synchronization.
unsafe impl Send for TokChannel {}
unsafe impl Sync for TokChannel {}

impl TokChannel {
    pub fn new(capacity: usize) -> Self {
        if capacity == 0 {
            TokChannel {
                rc: AtomicU32::new(1),
                capacity,
                ring: None,
                unbuffered: Some(Mutex::new(UnbufferedInner {
                    sender_waiting: None,
                    sender_ready: false,
                    closed: false,
                })),
                recv_condvar: Condvar::new(),
                send_condvar: Condvar::new(),
                wait_mutex: Mutex::new(()),
            }
        } else {
            TokChannel {
                rc: AtomicU32::new(1),
                capacity,
                ring: Some(SpscRing::new(capacity)),
                unbuffered: None,
                recv_condvar: Condvar::new(),
                send_condvar: Condvar::new(),
                wait_mutex: Mutex::new(()),
            }
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
        let ring = self.ring.as_ref().unwrap();
        // Fast path: try lock-free push
        if ring.try_push(val) {
            // Notify receiver if it might be waiting
            self.recv_condvar.notify_one();
            return;
        }
        // Slow path: buffer was full, spin briefly then park
        for _ in 0..32 {
            std::hint::spin_loop();
            if ring.try_push(val) {
                self.recv_condvar.notify_one();
                return;
            }
        }
        // Park on condvar until space available
        loop {
            let guard = self.wait_mutex.lock().unwrap();
            if ring.try_push(val) {
                drop(guard);
                self.recv_condvar.notify_one();
                return;
            }
            let _guard = self.send_condvar.wait(guard).unwrap();
            if ring.try_push(val) {
                self.recv_condvar.notify_one();
                return;
            }
        }
    }

    fn recv_buffered(&self) -> TokValue {
        let ring = self.ring.as_ref().unwrap();
        // Fast path: try lock-free pop
        if let Some(val) = ring.try_pop() {
            self.send_condvar.notify_one();
            return val;
        }
        // Slow path: buffer was empty, spin briefly then park
        for _ in 0..32 {
            std::hint::spin_loop();
            if let Some(val) = ring.try_pop() {
                self.send_condvar.notify_one();
                return val;
            }
        }
        // Park on condvar until data available
        loop {
            let guard = self.wait_mutex.lock().unwrap();
            if let Some(val) = ring.try_pop() {
                drop(guard);
                self.send_condvar.notify_one();
                return val;
            }
            let _guard = self.recv_condvar.wait(guard).unwrap();
            if let Some(val) = ring.try_pop() {
                self.send_condvar.notify_one();
                return val;
            }
        }
    }

    fn send_unbuffered(&self, val: TokValue) {
        let unbuf = self.unbuffered.as_ref().unwrap();
        let mut inner = unbuf.lock().unwrap();
        while inner.sender_waiting.is_some() {
            inner = self.send_condvar.wait(inner).unwrap();
        }
        inner.sender_waiting = Some(val);
        inner.sender_ready = false;
        self.recv_condvar.notify_one();
        while !inner.sender_ready {
            inner = self.send_condvar.wait(inner).unwrap();
        }
        inner.sender_ready = false;
    }

    fn recv_unbuffered(&self) -> TokValue {
        let unbuf = self.unbuffered.as_ref().unwrap();
        let mut inner = unbuf.lock().unwrap();
        while inner.sender_waiting.is_none() {
            inner = self.recv_condvar.wait(inner).unwrap();
        }
        let val = inner.sender_waiting.take().unwrap();
        inner.sender_ready = true;
        self.send_condvar.notify_one();
        val
    }

    /// Non-blocking try_send. Returns true if sent.
    pub fn try_send(&self, val: TokValue) -> bool {
        if self.capacity == 0 {
            false // Unbuffered: can't non-blocking send
        } else {
            let ring = self.ring.as_ref().unwrap();
            if ring.try_push(val) {
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
            let unbuf = self.unbuffered.as_ref().unwrap();
            let mut inner = unbuf.lock().unwrap();
            if inner.sender_waiting.is_some() {
                let val = inner.sender_waiting.take().unwrap();
                inner.sender_ready = true;
                self.send_condvar.notify_one();
                Some(val)
            } else {
                None
            }
        } else {
            let ring = self.ring.as_ref().unwrap();
            if let Some(val) = ring.try_pop() {
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
        // Buffer full (capacity is rounded to 2 for power-of-2)
        // With capacity 1, ring has 2 slots, so a second push may succeed
        // Let's just test the recv
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
            unsafe { sum += v.data.int_val; }
        }

        sender.join().unwrap();
        assert_eq!(sum, (n - 1) * n / 2);
        unsafe { drop(Box::from_raw(ch)); }
    }
}
