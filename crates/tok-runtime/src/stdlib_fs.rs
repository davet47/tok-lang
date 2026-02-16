//! Standard library: `@"fs"` module.
//!
//! Provides filesystem functions: fread, fwrite, fappend, fexists, fls, fmk, frm.

use crate::array::TokArray;
use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{TokValue, TAG_STRING};

use std::fs;

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

/// fread(path) -> Str
#[no_mangle]
pub extern "C" fn tok_fs_fread_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        match fs::read_to_string(path) {
            Ok(contents) => TokValue::from_string(TokString::alloc(contents)),
            Err(e) => {
                eprintln!("fs.fread error: {}", e);
                TokValue::nil()
            }
        }
    }
}

/// fwrite(path, content) -> Nil
#[no_mangle]
pub extern "C" fn tok_fs_fwrite_t(
    _env: *mut u8,
    tag1: i64, data1: i64,
    tag2: i64, data2: i64,
) -> TokValue {
    unsafe {
        let path = arg_to_str(tag1, data1).to_string();
        let content = arg_to_str(tag2, data2);
        if let Err(e) = fs::write(&path, content) {
            eprintln!("fs.fwrite error: {}", e);
        }
    }
    TokValue::nil()
}

/// fappend(path, content) -> Nil
#[no_mangle]
pub extern "C" fn tok_fs_fappend_t(
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
                    eprintln!("fs.fappend error: {}", e);
                }
            }
            Err(e) => eprintln!("fs.fappend error: {}", e),
        }
    }
    TokValue::nil()
}

/// fexists(path) -> Bool
#[no_mangle]
pub extern "C" fn tok_fs_fexists_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        TokValue::from_bool(std::path::Path::new(path).exists())
    }
}

/// fls(path) -> Array<Str>
#[no_mangle]
pub extern "C" fn tok_fs_fls_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
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
            Err(e) => eprintln!("fs.fls error: {}", e),
        }
        TokValue::from_array(arr)
    }
}

/// fmk(path) -> Nil (recursive mkdir)
#[no_mangle]
pub extern "C" fn tok_fs_fmk_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        if let Err(e) = fs::create_dir_all(path) {
            eprintln!("fs.fmk error: {}", e);
        }
    }
    TokValue::nil()
}

/// frm(path) -> Nil
#[no_mangle]
pub extern "C" fn tok_fs_frm_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let path = arg_to_str(tag, data);
        let p = std::path::Path::new(path);
        let result = if p.is_dir() {
            fs::remove_dir_all(path)
        } else {
            fs::remove_file(path)
        };
        if let Err(e) = result {
            eprintln!("fs.frm error: {}", e);
        }
    }
    TokValue::nil()
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
pub extern "C" fn tok_stdlib_fs() -> *mut TokMap {
    let m = TokMap::alloc();

    // 1-arg functions
    insert_func(m, "fread",   tok_fs_fread_t   as *const u8, 1);
    insert_func(m, "fexists", tok_fs_fexists_t as *const u8, 1);
    insert_func(m, "fls",     tok_fs_fls_t     as *const u8, 1);
    insert_func(m, "fmk",     tok_fs_fmk_t     as *const u8, 1);
    insert_func(m, "frm",     tok_fs_frm_t     as *const u8, 1);

    // 2-arg functions
    insert_func(m, "fwrite",  tok_fs_fwrite_t  as *const u8, 2);
    insert_func(m, "fappend", tok_fs_fappend_t as *const u8, 2);

    m
}
