//! Standard library: `@"re"` module.
//!
//! Provides regex functions: rmatch, rfind, rall, rsub.

use crate::array::TokArray;
use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{TokValue, TAG_STRING};

use regex::Regex;

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

/// rmatch(s, pat) -> Bool
/// Test if string matches the regex pattern.
#[no_mangle]
pub extern "C" fn tok_re_rmatch_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let s = arg_to_str(tag1, data1);
        let pat = arg_to_str(tag2, data2);
        match Regex::new(pat) {
            Ok(re) => TokValue::from_bool(re.is_match(s)),
            Err(e) => {
                eprintln!("re.rmatch error: {}", e);
                TokValue::from_bool(false)
            }
        }
    }
}

/// rfind(s, pat) -> Str | Nil
/// Find first match in string. Returns matched text or Nil.
#[no_mangle]
pub extern "C" fn tok_re_rfind_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let s = arg_to_str(tag1, data1);
        let pat = arg_to_str(tag2, data2);
        match Regex::new(pat) {
            Ok(re) => match re.find(s) {
                Some(m) => TokValue::from_string(TokString::alloc(m.as_str().to_string())),
                None => TokValue::nil(),
            },
            Err(e) => {
                eprintln!("re.rfind error: {}", e);
                TokValue::nil()
            }
        }
    }
}

/// rall(s, pat) -> Array<Str>
/// Find all matches in string. Returns array of matched strings.
#[no_mangle]
pub extern "C" fn tok_re_rall_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let s = arg_to_str(tag1, data1);
        let pat = arg_to_str(tag2, data2);
        let arr = TokArray::alloc();
        match Regex::new(pat) {
            Ok(re) => {
                for m in re.find_iter(s) {
                    (*arr).data.push(TokValue::from_string(TokString::alloc(
                        m.as_str().to_string(),
                    )));
                }
            }
            Err(e) => {
                eprintln!("re.rall error: {}", e);
            }
        }
        TokValue::from_array(arr)
    }
}

/// rsub(s, pat, replacement) -> Str
/// Replace all matches of pattern with replacement string.
#[no_mangle]
pub extern "C" fn tok_re_rsub_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
    tag3: i64,
    data3: i64,
) -> TokValue {
    unsafe {
        let s = arg_to_str(tag1, data1);
        let pat = arg_to_str(tag2, data2);
        let rep = arg_to_str(tag3, data3);
        match Regex::new(pat) {
            Ok(re) => {
                let result = re.replace_all(s, rep).to_string();
                TokValue::from_string(TokString::alloc(result))
            }
            Err(e) => {
                eprintln!("re.rsub error: {}", e);
                TokValue::from_string(TokString::alloc(s.to_string()))
            }
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
pub extern "C" fn tok_stdlib_re() -> *mut TokMap {
    let m = TokMap::alloc();

    insert_func(m, "rmatch", tok_re_rmatch_t as *const u8, 2);
    insert_func(m, "rfind", tok_re_rfind_t as *const u8, 2);
    insert_func(m, "rall", tok_re_rall_t as *const u8, 2);
    insert_func(m, "rsub", tok_re_rsub_t as *const u8, 3);

    m
}
