//! Standard library: `@"re"` module.
//!
//! Provides regex functions: rmatch, rfind, rall, rsub.
//! Compiled regexes are cached in a thread-local LRU cache to avoid
//! recompiling the same pattern on every call.

use crate::array::TokArray;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::TokValue;

use regex::Regex;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::stdlib_helpers::{arg_to_str, insert_func};

// ═══════════════════════════════════════════════════════════════
// Regex cache
// ═══════════════════════════════════════════════════════════════

/// Maximum number of cached compiled regexes per thread.
const REGEX_CACHE_CAPACITY: usize = 64;

thread_local! {
    static REGEX_CACHE: RefCell<HashMap<String, Regex>> = RefCell::new(HashMap::new());
}

/// Get a compiled Regex from the cache, or compile and cache it.
/// Returns None if the pattern is invalid.
fn get_or_compile_regex(pat: &str) -> Option<Regex> {
    REGEX_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(re) = cache.get(pat) {
            return Some(re.clone());
        }
        match Regex::new(pat) {
            Ok(re) => {
                // Evict oldest entries if at capacity
                if cache.len() >= REGEX_CACHE_CAPACITY {
                    // Simple eviction: clear half the cache
                    let keys: Vec<String> = cache.keys().take(REGEX_CACHE_CAPACITY / 2).cloned().collect();
                    for k in keys {
                        cache.remove(&k);
                    }
                }
                cache.insert(pat.to_string(), re.clone());
                Some(re)
            }
            Err(_) => None,
        }
    })
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
        match get_or_compile_regex(pat) {
            Some(re) => TokValue::from_bool(re.is_match(s)),
            None => {
                eprintln!("re.rmatch error: invalid pattern '{}'", pat);
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
        match get_or_compile_regex(pat) {
            Some(re) => match re.find(s) {
                Some(m) => TokValue::from_string(TokString::alloc(m.as_str().to_string())),
                None => TokValue::nil(),
            },
            None => {
                eprintln!("re.rfind error: invalid pattern '{}'", pat);
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
        match get_or_compile_regex(pat) {
            Some(re) => {
                for m in re.find_iter(s) {
                    (*arr).data.push(TokValue::from_string(TokString::alloc(
                        m.as_str().to_string(),
                    )));
                }
            }
            None => {
                eprintln!("re.rall error: invalid pattern '{}'", pat);
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
        match get_or_compile_regex(pat) {
            Some(re) => {
                let result = re.replace_all(s, rep).to_string();
                TokValue::from_string(TokString::alloc(result))
            }
            None => {
                eprintln!("re.rsub error: invalid pattern '{}'", pat);
                TokValue::from_string(TokString::alloc(s.to_string()))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Module constructor
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_stdlib_re() -> *mut TokMap {
    let m = TokMap::alloc();

    insert_func(m, "rmatch", tok_re_rmatch_t as *const u8, 2);
    insert_func(m, "rfind", tok_re_rfind_t as *const u8, 2);
    insert_func(m, "rall", tok_re_rall_t as *const u8, 2);
    insert_func(m, "rsub", tok_re_rsub_t as *const u8, 3);

    m
}
