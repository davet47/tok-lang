//! Standard library: `@"json"` module.
//!
//! Provides JSON parsing and serialization.

use crate::array::TokArray;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{
    TokValue, TAG_ARRAY, TAG_BOOL, TAG_FLOAT, TAG_FUNC, TAG_INT, TAG_MAP, TAG_NIL, TAG_STRING,
    TAG_TUPLE,
};

use serde_json::Value as JsonValue;

use crate::stdlib_helpers::{arg_to_str, insert_func};

const MAX_DEPTH: usize = 128;

/// Convert a serde_json::Value into a TokValue.
fn json_to_tok(jv: &JsonValue) -> TokValue {
    json_to_tok_depth(jv, 0)
}

fn json_to_tok_depth(jv: &JsonValue, depth: usize) -> TokValue {
    if depth >= MAX_DEPTH {
        return TokValue::nil();
    }
    match jv {
        JsonValue::Null => TokValue::nil(),
        JsonValue::Bool(b) => TokValue::from_bool(*b),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                TokValue::from_int(i)
            } else if let Some(f) = n.as_f64() {
                TokValue::from_float(f)
            } else {
                TokValue::nil()
            }
        }
        JsonValue::String(s) => TokValue::from_string(TokString::alloc(s.clone())),
        JsonValue::Array(arr) => {
            let tok_arr = TokArray::alloc();
            unsafe {
                for item in arr {
                    (*tok_arr).data.push(json_to_tok_depth(item, depth + 1));
                }
            }
            TokValue::from_array(tok_arr)
        }
        JsonValue::Object(obj) => {
            let tok_map = TokMap::alloc();
            unsafe {
                for (k, v) in obj {
                    (*tok_map).data.insert(k.clone(), json_to_tok_depth(v, depth + 1));
                }
            }
            TokValue::from_map(tok_map)
        }
    }
}

/// Convert a TokValue into a serde_json::Value.
unsafe fn tok_to_json(tv: &TokValue) -> JsonValue {
    tok_to_json_depth(tv, 0)
}

unsafe fn tok_to_json_depth(tv: &TokValue, depth: usize) -> JsonValue {
    if depth >= MAX_DEPTH {
        return JsonValue::Null;
    }
    match tv.tag {
        TAG_NIL => JsonValue::Null,
        TAG_BOOL => JsonValue::Bool(tv.data.bool_val != 0),
        TAG_INT => JsonValue::Number(serde_json::Number::from(tv.data.int_val)),
        TAG_FLOAT => {
            let f = tv.data.float_val;
            if let Some(n) = serde_json::Number::from_f64(f) {
                JsonValue::Number(n)
            } else {
                JsonValue::Null // NaN/Infinity can't be represented in JSON
            }
        }
        TAG_STRING => {
            let p = tv.data.string_ptr;
            if p.is_null() {
                JsonValue::Null
            } else {
                JsonValue::String((*p).data.clone())
            }
        }
        TAG_ARRAY => {
            let p = tv.data.array_ptr;
            if p.is_null() {
                JsonValue::Array(vec![])
            } else {
                let items: Vec<JsonValue> = (*p)
                    .data
                    .iter()
                    .map(|v| tok_to_json_depth(v, depth + 1))
                    .collect();
                JsonValue::Array(items)
            }
        }
        TAG_MAP => {
            let p = tv.data.map_ptr;
            if p.is_null() {
                JsonValue::Object(serde_json::Map::new())
            } else {
                let mut obj = serde_json::Map::new();
                for (k, v) in &(*p).data {
                    obj.insert(k.clone(), tok_to_json_depth(v, depth + 1));
                }
                JsonValue::Object(obj)
            }
        }
        TAG_TUPLE => {
            let p = tv.data.tuple_ptr;
            if p.is_null() {
                JsonValue::Array(vec![])
            } else {
                let items: Vec<JsonValue> = (*p)
                    .data
                    .iter()
                    .map(|v| tok_to_json_depth(v, depth + 1))
                    .collect();
                JsonValue::Array(items)
            }
        }
        TAG_FUNC => JsonValue::String("<function>".to_string()),
        _ => JsonValue::Null,
    }
}

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

/// parse(json_string) -> Any
#[no_mangle]
pub extern "C" fn tok_json_parse_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        match serde_json::from_str::<JsonValue>(s) {
            Ok(jv) => json_to_tok(&jv),
            Err(e) => {
                eprintln!("json.parse error: {}", e);
                TokValue::nil()
            }
        }
    }
}

/// stringify(value) -> Str
#[no_mangle]
pub extern "C" fn tok_json_stringify_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let tv = TokValue::from_tag_data(tag, data);
        let jv = tok_to_json(&tv);
        let s = serde_json::to_string(&jv).unwrap_or_else(|_| "null".to_string());
        TokValue::from_string(TokString::alloc(s))
    }
}

/// pretty(value) -> Str (pretty-printed JSON)
#[no_mangle]
pub extern "C" fn tok_json_pretty_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let tv = TokValue::from_tag_data(tag, data);
        let jv = tok_to_json(&tv);
        let s = serde_json::to_string_pretty(&jv).unwrap_or_else(|_| "null".to_string());
        TokValue::from_string(TokString::alloc(s))
    }
}

// ═══════════════════════════════════════════════════════════════
// Module constructor
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_stdlib_json() -> *mut TokMap {
    let m = TokMap::alloc();

    // New names (spec v0.1)
    insert_func(m, "jparse", tok_json_parse_t as *const u8, 1);
    insert_func(m, "jstr", tok_json_stringify_t as *const u8, 1);
    insert_func(m, "jpretty", tok_json_pretty_t as *const u8, 1);
    // Legacy names for backward compatibility
    insert_func(m, "parse", tok_json_parse_t as *const u8, 1);
    insert_func(m, "stringify", tok_json_stringify_t as *const u8, 1);
    insert_func(m, "pretty", tok_json_pretty_t as *const u8, 1);

    m
}
