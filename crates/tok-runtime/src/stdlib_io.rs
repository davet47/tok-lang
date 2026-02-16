//! Standard library: `@"io"` module.
//!
//! Provides file I/O and filesystem functions.

use crate::array::TokArray;
use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{TokValue, TAG_STRING};

use std::fs;
use std::io::{self, BufRead};

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

#[inline]
unsafe fn arg_to_str<'a>(tag: i64, data: i64) -> &'a str {
    if tag as u8 == TAG_STRING {
        let ptr = data as *mut TokString;
        if !ptr.is_null() {
            return &(*ptr).data;
        }
    }
    ""
}

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

/// read_file(path) -> Str
#[no_mangle]
pub extern "C" fn tok_io_read_file_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        match fs::read_to_string(path) {
            Ok(contents) => TokValue::from_string(TokString::alloc(contents)),
            Err(e) => {
                eprintln!("io.read_file error: {}", e);
                TokValue::nil()
            }
        }
    }
}

/// write_file(path, content) -> Nil
#[no_mangle]
pub extern "C" fn tok_io_write_file_t(
    _env: *mut u8,
    tag1: i64, data1: i64,
    tag2: i64, data2: i64,
) -> TokValue {
    unsafe {
        let path = arg_to_str(tag1, data1).to_string();
        let content = arg_to_str(tag2, data2);
        if let Err(e) = fs::write(&path, content) {
            eprintln!("io.write_file error: {}", e);
        }
    }
    TokValue::nil()
}

/// append_file(path, content) -> Nil
#[no_mangle]
pub extern "C" fn tok_io_append_file_t(
    _env: *mut u8,
    tag1: i64, data1: i64,
    tag2: i64, data2: i64,
) -> TokValue {
    unsafe {
        let path = arg_to_str(tag1, data1).to_string();
        let content = arg_to_str(tag2, data2);
        use std::io::Write;
        match fs::OpenOptions::new().append(true).create(true).open(&path) {
            Ok(mut file) => {
                if let Err(e) = file.write_all(content.as_bytes()) {
                    eprintln!("io.append_file error: {}", e);
                }
            }
            Err(e) => eprintln!("io.append_file error: {}", e),
        }
    }
    TokValue::nil()
}

/// exists(path) -> Bool
#[no_mangle]
pub extern "C" fn tok_io_exists_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        TokValue::from_bool(std::path::Path::new(path).exists())
    }
}

/// mkdir(path) -> Nil (recursive)
#[no_mangle]
pub extern "C" fn tok_io_mkdir_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        if let Err(e) = fs::create_dir_all(path) {
            eprintln!("io.mkdir error: {}", e);
        }
    }
    TokValue::nil()
}

/// ls(path) -> Array<Str>
#[no_mangle]
pub extern "C" fn tok_io_ls_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        let arr = TokArray::alloc();
        match fs::read_dir(path) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    (*arr).data.push(TokValue::from_string(TokString::alloc(name)));
                }
            }
            Err(e) => eprintln!("io.ls error: {}", e),
        }
        TokValue::from_array(arr)
    }
}

/// rm(path) -> Nil
#[no_mangle]
pub extern "C" fn tok_io_rm_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        let p = std::path::Path::new(path);
        let result = if p.is_dir() {
            fs::remove_dir_all(path)
        } else {
            fs::remove_file(path)
        };
        if let Err(e) = result {
            eprintln!("io.rm error: {}", e);
        }
    }
    TokValue::nil()
}

/// read_line() -> Str
#[no_mangle]
pub extern "C" fn tok_io_read_line_t(_env: *mut u8) -> TokValue {
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
            eprintln!("io.read_line error: {}", e);
            TokValue::nil()
        }
    }
}

/// is_dir(path) -> Bool
#[no_mangle]
pub extern "C" fn tok_io_is_dir_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        TokValue::from_bool(std::path::Path::new(path).is_dir())
    }
}

/// is_file(path) -> Bool
#[no_mangle]
pub extern "C" fn tok_io_is_file_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        TokValue::from_bool(std::path::Path::new(path).is_file())
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

    // 0-arg functions
    insert_func(m, "read_line", tok_io_read_line_t as *const u8, 0);

    // 1-arg functions
    insert_func(m, "read_file",  tok_io_read_file_t  as *const u8, 1);
    insert_func(m, "exists",     tok_io_exists_t     as *const u8, 1);
    insert_func(m, "mkdir",      tok_io_mkdir_t      as *const u8, 1);
    insert_func(m, "ls",         tok_io_ls_t         as *const u8, 1);
    insert_func(m, "rm",         tok_io_rm_t         as *const u8, 1);
    insert_func(m, "is_dir",     tok_io_is_dir_t     as *const u8, 1);
    insert_func(m, "is_file",    tok_io_is_file_t    as *const u8, 1);

    // 2-arg functions
    insert_func(m, "write_file",  tok_io_write_file_t  as *const u8, 2);
    insert_func(m, "append_file", tok_io_append_file_t as *const u8, 2);

    m
}
