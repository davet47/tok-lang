//! Standard library: `@"str"` module.
//!
//! Provides string manipulation functions as a TokMap of closures.

use crate::array::TokArray;
use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{TokValue, TAG_INT, TAG_STRING};

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

#[inline]
fn arg_to_i64(tag: i64, data: i64) -> i64 {
    if tag as u8 == TAG_INT {
        data
    } else {
        0
    }
}

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

// --- 1-arg Str -> Str ---

#[no_mangle]
pub extern "C" fn tok_str_upper_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        TokValue::from_string(TokString::alloc(s.to_uppercase()))
    }
}

#[no_mangle]
pub extern "C" fn tok_str_lower_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        TokValue::from_string(TokString::alloc(s.to_lowercase()))
    }
}

#[no_mangle]
pub extern "C" fn tok_str_trim_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        TokValue::from_string(TokString::alloc(s.trim().to_string()))
    }
}

#[no_mangle]
pub extern "C" fn tok_str_trim_left_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        TokValue::from_string(TokString::alloc(s.trim_start().to_string()))
    }
}

#[no_mangle]
pub extern "C" fn tok_str_trim_right_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        TokValue::from_string(TokString::alloc(s.trim_end().to_string()))
    }
}

#[no_mangle]
pub extern "C" fn tok_str_chars_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        let arr = TokArray::alloc();
        for ch in s.chars() {
            let ch_str = TokString::alloc(ch.to_string());
            (*arr).data.push(TokValue::from_string(ch_str));
        }
        TokValue::from_array(arr)
    }
}

#[no_mangle]
pub extern "C" fn tok_str_bytes_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        let arr = TokArray::alloc();
        for b in s.bytes() {
            (*arr).data.push(TokValue::from_int(b as i64));
        }
        TokValue::from_array(arr)
    }
}

#[no_mangle]
pub extern "C" fn tok_str_rev_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        TokValue::from_string(TokString::alloc(s.chars().rev().collect()))
    }
}

// --- 1-arg Str -> Int ---

#[no_mangle]
pub extern "C" fn tok_str_len_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        TokValue::from_int(s.chars().count() as i64)
    }
}

// --- 2-arg Str, Str -> Bool ---

#[no_mangle]
pub extern "C" fn tok_str_contains_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let s = arg_to_str(tag1, data1);
        let sub = arg_to_str(tag2, data2);
        TokValue::from_bool(s.contains(sub))
    }
}

#[no_mangle]
pub extern "C" fn tok_str_starts_with_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let s = arg_to_str(tag1, data1);
        let prefix = arg_to_str(tag2, data2);
        TokValue::from_bool(s.starts_with(prefix))
    }
}

#[no_mangle]
pub extern "C" fn tok_str_ends_with_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let s = arg_to_str(tag1, data1);
        let suffix = arg_to_str(tag2, data2);
        TokValue::from_bool(s.ends_with(suffix))
    }
}

// --- 2-arg Str, Str -> Int ---

#[no_mangle]
pub extern "C" fn tok_str_index_of_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let s = arg_to_str(tag1, data1);
        let needle = arg_to_str(tag2, data2);
        match s.find(needle) {
            Some(pos) => TokValue::from_int(pos as i64),
            None => TokValue::from_int(-1),
        }
    }
}

// --- 2-arg Str, Int -> Str ---

#[no_mangle]
pub extern "C" fn tok_str_repeat_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let s = arg_to_str(tag1, data1);
        let n = arg_to_i64(tag2, data2).max(0) as usize;
        TokValue::from_string(TokString::alloc(s.repeat(n)))
    }
}

// --- 2-arg Str, Str -> Array ---

#[no_mangle]
pub extern "C" fn tok_str_split_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let s = arg_to_str(tag1, data1);
        let delim = arg_to_str(tag2, data2);
        let arr = TokArray::alloc();
        for part in s.split(delim) {
            let part_str = TokString::alloc(part.to_string());
            (*arr).data.push(TokValue::from_string(part_str));
        }
        TokValue::from_array(arr)
    }
}

// --- 3-arg Str, Str, Str -> Str ---

#[no_mangle]
pub extern "C" fn tok_str_replace_t(
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
        let from = arg_to_str(tag2, data2);
        let to = arg_to_str(tag3, data3);
        TokValue::from_string(TokString::alloc(s.replace(from, to)))
    }
}

// --- 3-arg Str, Int, Str -> Str (pad) ---

#[no_mangle]
pub extern "C" fn tok_str_pad_left_t(
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
        let width = arg_to_i64(tag2, data2).max(0) as usize;
        let pad = arg_to_str(tag3, data3);
        let pad_char = pad.chars().next().unwrap_or(' ');
        let char_count = s.chars().count();
        if char_count >= width {
            TokValue::from_string(TokString::alloc(s.to_string()))
        } else {
            let padding: String = std::iter::repeat_n(pad_char, width - char_count).collect();
            TokValue::from_string(TokString::alloc(format!("{}{}", padding, s)))
        }
    }
}

#[no_mangle]
pub extern "C" fn tok_str_pad_right_t(
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
        let width = arg_to_i64(tag2, data2).max(0) as usize;
        let pad = arg_to_str(tag3, data3);
        let pad_char = pad.chars().next().unwrap_or(' ');
        let char_count = s.chars().count();
        if char_count >= width {
            TokValue::from_string(TokString::alloc(s.to_string()))
        } else {
            let padding: String = std::iter::repeat_n(pad_char, width - char_count).collect();
            TokValue::from_string(TokString::alloc(format!("{}{}", s, padding)))
        }
    }
}

// --- 3-arg Str, Int, Int -> Str (substring) ---

#[no_mangle]
pub extern "C" fn tok_str_substr_t(
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
        let start = arg_to_i64(tag2, data2).max(0) as usize;
        let end = arg_to_i64(tag3, data3).max(0) as usize;
        let result: String = s
            .chars()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect();
        TokValue::from_string(TokString::alloc(result))
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
pub extern "C" fn tok_stdlib_str() -> *mut TokMap {
    let m = TokMap::alloc();

    // 1-arg functions
    insert_func(m, "upper", tok_str_upper_t as *const u8, 1);
    insert_func(m, "lower", tok_str_lower_t as *const u8, 1);
    insert_func(m, "trim", tok_str_trim_t as *const u8, 1);
    insert_func(m, "trim_left", tok_str_trim_left_t as *const u8, 1);
    insert_func(m, "trim_right", tok_str_trim_right_t as *const u8, 1);
    insert_func(m, "chars", tok_str_chars_t as *const u8, 1);
    insert_func(m, "bytes", tok_str_bytes_t as *const u8, 1);
    insert_func(m, "rev", tok_str_rev_t as *const u8, 1);
    insert_func(m, "len", tok_str_len_t as *const u8, 1);

    // 2-arg functions
    insert_func(m, "contains", tok_str_contains_t as *const u8, 2);
    insert_func(m, "starts_with", tok_str_starts_with_t as *const u8, 2);
    insert_func(m, "ends_with", tok_str_ends_with_t as *const u8, 2);
    insert_func(m, "index_of", tok_str_index_of_t as *const u8, 2);
    insert_func(m, "repeat", tok_str_repeat_t as *const u8, 2);
    insert_func(m, "split", tok_str_split_t as *const u8, 2);

    // 3-arg functions
    insert_func(m, "replace", tok_str_replace_t as *const u8, 3);
    insert_func(m, "pad_left", tok_str_pad_left_t as *const u8, 3);
    insert_func(m, "pad_right", tok_str_pad_right_t as *const u8, 3);
    insert_func(m, "substr", tok_str_substr_t as *const u8, 3);

    m
}
