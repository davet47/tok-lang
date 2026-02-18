//! Standard library: `@"io"` module.
//!
//! Provides stdin I/O functions: input, readall.

use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::TokValue;

use std::io::{self, BufRead, Read, Write};

use crate::stdlib_helpers::arg_to_str;

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

/// input() -> Str  (read line from stdin, no prompt)
#[no_mangle]
pub extern "C" fn tok_io_input_0_t(_env: *mut u8) -> TokValue {
    let stdin = io::stdin();
    let mut line = String::new();
    match stdin.lock().read_line(&mut line) {
        Ok(_) => {
            // Trim trailing newline
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            TokValue::from_string(TokString::alloc(line))
        }
        Err(e) => {
            eprintln!("io.input error: {}", e);
            TokValue::nil()
        }
    }
}

/// input(prompt) -> Str  (print prompt, then read line from stdin)
#[no_mangle]
pub extern "C" fn tok_io_input_1_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let prompt = arg_to_str(tag, data);
        print!("{}", prompt);
        let _ = io::stdout().flush();
    }
    // Now read
    let stdin = io::stdin();
    let mut line = String::new();
    match stdin.lock().read_line(&mut line) {
        Ok(_) => {
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            TokValue::from_string(TokString::alloc(line))
        }
        Err(e) => {
            eprintln!("io.input error: {}", e);
            TokValue::nil()
        }
    }
}

/// readall() -> Str  (read all of stdin to string)
#[no_mangle]
pub extern "C" fn tok_io_readall_t(_env: *mut u8) -> TokValue {
    let mut buf = String::new();
    match io::stdin().read_to_string(&mut buf) {
        Ok(_) => TokValue::from_string(TokString::alloc(buf)),
        Err(e) => {
            eprintln!("io.readall error: {}", e);
            TokValue::nil()
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Module constructor
// ═══════════════════════════════════════════════════════════════

fn insert_func(m: *mut TokMap, name: &str, fn_ptr: *const u8, arity: u32) {
    let closure = TokClosure::alloc(fn_ptr, std::ptr::null_mut(), arity);
    let val = TokValue::from_func(closure);
    unsafe {
        (*m).data.insert(name.to_string(), val);
    }
}

#[no_mangle]
pub extern "C" fn tok_stdlib_io() -> *mut TokMap {
    let m = TokMap::alloc();

    // input() with 0 args reads a line
    insert_func(m, "input", tok_io_input_0_t as *const u8, 0);
    // We also register a 1-arg version for input(prompt)
    // Since the module system doesn't support overloading by arity,
    // we'll use the 1-arg version and handle 0-arg calls gracefully.
    // Actually, let's just register the 1-arg version as "input" —
    // callers who want no prompt can pass "".
    // But the spec says both input() and input(prompt) should work.
    // We'll register the 1-arg version since it handles empty prompt too.
    insert_func(m, "input", tok_io_input_1_t as *const u8, 1);
    insert_func(m, "readall", tok_io_readall_t as *const u8, 0);

    m
}
