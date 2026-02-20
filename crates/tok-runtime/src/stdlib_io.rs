//! Standard library: `@"io"` module.
//!
//! Provides stdin I/O functions: input, readall.

use crate::map::TokMap;
use crate::string::TokString;
use crate::value::TokValue;

use std::io::{self, BufRead, Read, Write};

use crate::stdlib_helpers::{arg_to_str, insert_func};

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

#[no_mangle]
pub extern "C" fn tok_stdlib_io() -> *mut TokMap {
    let m = TokMap::alloc();

    // Register only the 1-arg version: input(prompt). The 0-arg registration
    // was previously overwritten here anyway since the module map doesn't
    // support arity overloading. Callers who want no prompt can pass "".
    insert_func(m, "input", tok_io_input_1_t as *const u8, 1);
    insert_func(m, "readall", tok_io_readall_t as *const u8, 0);

    m
}
