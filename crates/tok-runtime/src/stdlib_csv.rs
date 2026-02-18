//! Standard library: `@"csv"` module.
//!
//! Provides CSV parsing and serialization.
//! - `cparse(string)` — parse CSV string → array of maps (first row = headers)
//! - `cstr(value)` — encode array of maps → CSV string

use crate::array::TokArray;
use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{
    TokValue, TAG_ARRAY, TAG_BOOL, TAG_FLOAT, TAG_INT, TAG_MAP, TAG_NIL, TAG_STRING,
};

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
// CSV Parser
// ═══════════════════════════════════════════════════════════════

/// Parse a CSV string into a TokArray of TokMaps.
/// First row is treated as headers; subsequent rows become maps keyed by headers.
fn csv_parse(input: &str) -> TokValue {
    let rows = parse_csv_rows(input);
    if rows.is_empty() {
        return TokValue::from_array(TokArray::alloc());
    }

    let headers = &rows[0];
    let arr = TokArray::alloc();

    for row in &rows[1..] {
        let map = TokMap::alloc();
        for (i, header) in headers.iter().enumerate() {
            let val = if let Some(field) = row.get(i) {
                parse_field(field)
            } else {
                TokValue::nil()
            };
            unsafe {
                (*map).data.insert(header.clone(), val);
            }
        }
        unsafe {
            (*arr).data.push(TokValue::from_map(map));
        }
    }

    TokValue::from_array(arr)
}

/// Parse CSV input into a Vec of rows, each row a Vec of field strings.
/// Handles RFC 4180 quoting: fields in `"..."` can contain commas, newlines,
/// and `""` for escaped quotes.
fn parse_csv_rows(input: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                // Check for escaped quote ""
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current_field.push('"');
                } else {
                    // End of quoted field
                    in_quotes = false;
                }
            } else {
                current_field.push(ch);
            }
        } else {
            match ch {
                '"' => {
                    in_quotes = true;
                }
                ',' => {
                    current_row.push(std::mem::take(&mut current_field));
                }
                '\r' => {
                    // Handle \r\n
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    current_row.push(std::mem::take(&mut current_field));
                    if !current_row.is_empty() {
                        rows.push(std::mem::take(&mut current_row));
                    }
                }
                '\n' => {
                    current_row.push(std::mem::take(&mut current_field));
                    if !current_row.is_empty() {
                        rows.push(std::mem::take(&mut current_row));
                    }
                }
                _ => {
                    current_field.push(ch);
                }
            }
        }
    }

    // Final field and row
    current_row.push(current_field);
    // Only push if there's actual content (not just an empty trailing field from a trailing newline)
    if current_row.len() > 1 || !current_row[0].is_empty() {
        rows.push(current_row);
    }

    rows
}

/// Parse a single CSV field value, auto-detecting type.
fn parse_field(s: &str) -> TokValue {
    let trimmed = s.trim();
    match trimmed {
        "" => TokValue::nil(),
        "null" => TokValue::nil(),
        "true" => TokValue::from_bool(true),
        "false" => TokValue::from_bool(false),
        _ => {
            // Try integer
            if let Ok(i) = trimmed.parse::<i64>() {
                if !has_leading_zeros(trimmed) {
                    return TokValue::from_int(i);
                }
            }
            // Try float
            if let Ok(f) = trimmed.parse::<f64>() {
                if !has_leading_zeros(trimmed) && trimmed.contains('.') {
                    return TokValue::from_float(f);
                }
            }
            // String
            TokValue::from_string(TokString::alloc(trimmed.to_string()))
        }
    }
}

/// Check for leading zeros in a number string.
fn has_leading_zeros(s: &str) -> bool {
    let s = s.strip_prefix('-').unwrap_or(s);
    s.len() > 1 && s.starts_with('0') && !s.starts_with("0.")
}

// ═══════════════════════════════════════════════════════════════
// CSV Encoder
// ═══════════════════════════════════════════════════════════════

/// Convert an array of maps into a CSV string.
unsafe fn csv_stringify(tv: &TokValue) -> String {
    if tv.tag != TAG_ARRAY {
        return String::new();
    }
    let arr_ptr = tv.data.array_ptr;
    if arr_ptr.is_null() || (*arr_ptr).data.is_empty() {
        return String::new();
    }

    let data = &(*arr_ptr).data;

    // Get headers from the first map
    let first = &data[0];
    if first.tag != TAG_MAP {
        return String::new();
    }
    let first_map = first.data.map_ptr;
    if first_map.is_null() {
        return String::new();
    }
    let headers: Vec<String> = (*first_map).data.keys().cloned().collect();
    if headers.is_empty() {
        return String::new();
    }

    let mut out = String::new();

    // Write header row
    for (i, header) in headers.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        csv_quote_field(header, &mut out);
    }
    out.push('\n');

    // Write data rows
    for item in data {
        if item.tag != TAG_MAP {
            continue;
        }
        let map_ptr = item.data.map_ptr;
        if map_ptr.is_null() {
            continue;
        }
        for (i, header) in headers.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            if let Some(val) = (*map_ptr).data.get(header) {
                let s = value_to_csv_field(val);
                csv_quote_field(&s, &mut out);
            }
        }
        out.push('\n');
    }

    // Remove trailing newline
    if out.ends_with('\n') {
        out.pop();
    }

    out
}

/// Convert a TokValue to its CSV field string representation.
unsafe fn value_to_csv_field(tv: &TokValue) -> String {
    match tv.tag {
        TAG_NIL => String::new(),
        TAG_BOOL => {
            if tv.data.bool_val != 0 {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        TAG_INT => format!("{}", tv.data.int_val),
        TAG_FLOAT => format_float(tv.data.float_val),
        TAG_STRING => {
            let ptr = tv.data.string_ptr;
            if ptr.is_null() {
                String::new()
            } else {
                (*ptr).data.clone()
            }
        }
        _ => String::new(),
    }
}

/// Format a float for CSV output.
fn format_float(f: f64) -> String {
    if f == 0.0 {
        return "0".to_string();
    }
    let s = format!("{}", f);
    if s.contains('e') || s.contains('E') {
        let s = format!("{:.20}", f);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    } else {
        s
    }
}

/// Quote a CSV field if it contains special characters (comma, quote, newline).
/// Escapes `"` as `""` per RFC 4180.
fn csv_quote_field(s: &str, out: &mut String) {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        out.push('"');
        for ch in s.chars() {
            if ch == '"' {
                out.push_str("\"\"");
            } else {
                out.push(ch);
            }
        }
        out.push('"');
    } else {
        out.push_str(s);
    }
}

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

/// cparse(csv_string) -> Any
#[no_mangle]
pub extern "C" fn tok_csv_parse_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        csv_parse(s)
    }
}

/// cstr(value) -> Str
#[no_mangle]
pub extern "C" fn tok_csv_stringify_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let tv = TokValue::from_tag_data(tag, data);
        let s = csv_stringify(&tv);
        TokValue::from_string(TokString::alloc(s))
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
pub extern "C" fn tok_stdlib_csv() -> *mut TokMap {
    let m = TokMap::alloc();

    // Primary names
    insert_func(m, "cparse", tok_csv_parse_t as *const u8, 1);
    insert_func(m, "cstr", tok_csv_stringify_t as *const u8, 1);
    // Legacy names for consistency
    insert_func(m, "parse", tok_csv_parse_t as *const u8, 1);
    insert_func(m, "stringify", tok_csv_stringify_t as *const u8, 1);

    m
}
