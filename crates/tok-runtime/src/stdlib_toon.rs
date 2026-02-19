//! Standard library: `@"toon"` module.
//!
//! Provides TOON (Token-Oriented Object Notation) parsing and serialization.
//! TOON is a compact, JSON-compatible format optimized for LLM token efficiency.

use crate::array::TokArray;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{
    TokValue, TAG_ARRAY, TAG_BOOL, TAG_FLOAT, TAG_FUNC, TAG_INT, TAG_MAP, TAG_NIL, TAG_STRING,
    TAG_TUPLE,
};

use crate::stdlib_helpers::{arg_to_str, insert_func};

// ═══════════════════════════════════════════════════════════════
// TOON Parser
// ═══════════════════════════════════════════════════════════════

/// A parsed line with its indentation level and content.
struct Line<'a> {
    indent: usize,
    content: &'a str,
}

struct ToonParser<'a> {
    lines: Vec<Line<'a>>,
    pos: usize,
}

impl<'a> ToonParser<'a> {
    fn new(input: &'a str) -> Self {
        let lines = input
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let indent = l.len() - l.trim_start().len();
                Line {
                    indent,
                    content: l.trim_start(),
                }
            })
            .collect();
        ToonParser { lines, pos: 0 }
    }

    fn peek(&self) -> Option<&Line<'a>> {
        self.lines.get(self.pos)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn at_end(&self) -> bool {
        self.pos >= self.lines.len()
    }

    /// Top-level parse entry point.
    fn parse_root(&mut self) -> TokValue {
        if self.at_end() {
            return TokValue::from_map(TokMap::alloc());
        }

        // Root array: starts with `[`
        if self.lines[self.pos].content.starts_with('[') {
            return self.parse_root_array();
        }

        // Single primitive (one line, no key-value and no array header)
        if self.lines.len() == 1 {
            let content = self.lines[self.pos].content;
            if !line_is_key_value(content) && try_parse_keyed_array_header(content).is_none() {
                self.advance();
                return parse_primitive(content);
            }
        }

        // Otherwise: root object
        self.parse_object_block(0)
    }

    /// Parse a root-level array (no key prefix).
    fn parse_root_array(&mut self) -> TokValue {
        let line = self.lines[self.pos].content;
        let (_count, delim, fields) = parse_array_header_no_key(line);

        if let Some(fields) = fields {
            // Tabular root array
            let base_indent = self.lines[self.pos].indent;
            self.advance();
            self.parse_tabular_rows(&fields, delim, base_indent)
        } else {
            // Check for inline values after `:` (e.g. `[3]: x,y,z`)
            let after_colon = line_after_header_colon(line);
            if !after_colon.is_empty() {
                self.advance();
                parse_csv_primitives(after_colon, delim)
            } else {
                // Mixed root array
                let base_indent = self.lines[self.pos].indent;
                self.advance();
                self.parse_mixed_array(base_indent)
            }
        }
    }

    /// Parse an object block at a given indentation level.
    fn parse_object_block(&mut self, base_indent: usize) -> TokValue {
        let map = TokMap::alloc();

        while let Some(line) = self.peek() {
            if line.indent < base_indent {
                break;
            }
            if line.indent > base_indent {
                break; // shouldn't happen at top level, but guard
            }

            let content = line.content;

            // Try to parse as array header: key[N]... or key[N]{...}...
            if let Some((key, _count, delim, fields)) = try_parse_keyed_array_header(content) {
                if let Some(fields) = fields {
                    // Tabular array
                    let parent_indent = line.indent;
                    self.advance();
                    let arr = self.parse_tabular_rows(&fields, delim, parent_indent);
                    unsafe {
                        (*map).data.insert(key.to_string(), arr);
                    }
                } else {
                    // Could be inline primitive array or mixed array
                    let after_colon = line_after_header_colon(content);
                    if !after_colon.is_empty() {
                        // Inline: tags[3]: foo,bar,baz
                        self.advance();
                        let arr = parse_csv_primitives(after_colon, delim);
                        unsafe {
                            (*map).data.insert(key.to_string(), arr);
                        }
                    } else {
                        // Mixed array
                        let parent_indent = line.indent;
                        self.advance();
                        let arr = self.parse_mixed_array(parent_indent);
                        unsafe {
                            (*map).data.insert(key.to_string(), arr);
                        }
                    }
                }
                continue;
            }

            // Regular key: value
            if let Some((key, val_str)) = split_key_value(content) {
                self.advance();
                if val_str.is_empty() {
                    // Could be nested object or empty
                    if let Some(next) = self.peek() {
                        if next.indent > base_indent {
                            let child_indent = next.indent;
                            // Check if children are array items (start with "- ")
                            if next.content.starts_with("- ") {
                                let arr = self.parse_mixed_array(base_indent);
                                unsafe {
                                    (*map).data.insert(key.to_string(), arr);
                                }
                            } else {
                                let child = self.parse_object_block(child_indent);
                                unsafe {
                                    (*map).data.insert(key.to_string(), child);
                                }
                            }
                        } else {
                            // key: with nothing following at deeper indent -> nil
                            unsafe {
                                (*map).data.insert(key.to_string(), TokValue::nil());
                            }
                        }
                    } else {
                        unsafe {
                            (*map).data.insert(key.to_string(), TokValue::nil());
                        }
                    }
                } else {
                    let val = parse_primitive(val_str);
                    unsafe {
                        (*map).data.insert(key.to_string(), val);
                    }
                }
                continue;
            }

            // Unrecognized line — skip
            self.advance();
        }

        TokValue::from_map(map)
    }

    /// Parse tabular rows (CSV lines) following a `key[N]{f1,f2}:` header.
    fn parse_tabular_rows(
        &mut self,
        fields: &[String],
        delim: char,
        parent_indent: usize,
    ) -> TokValue {
        let arr = TokArray::alloc();
        while let Some(line) = self.peek() {
            if line.indent <= parent_indent {
                break;
            }
            let values = split_csv(line.content, delim);
            let row = TokMap::alloc();
            for (i, field) in fields.iter().enumerate() {
                let val = if let Some(v) = values.get(i) {
                    parse_primitive(v)
                } else {
                    TokValue::nil()
                };
                unsafe {
                    (*row).data.insert(field.clone(), val);
                }
            }
            unsafe {
                (*arr).data.push(TokValue::from_map(row));
            }
            self.advance();
        }
        TokValue::from_array(arr)
    }

    /// Parse a mixed array: children at indent > parent, each prefixed with `- `.
    fn parse_mixed_array(&mut self, parent_indent: usize) -> TokValue {
        let arr = TokArray::alloc();
        while let Some(line) = self.peek() {
            if line.indent <= parent_indent {
                break;
            }
            if let Some(rest) = line.content.strip_prefix("- ") {
                let item_indent = line.indent;

                // Check if the rest is a keyed array header (e.g. `- users[2]{id,name}:`)
                if let Some((_key, _count, _delim, _fields)) = try_parse_keyed_array_header(rest) {
                    // Complex list item — treat as an object starting from this "- " line
                    // We need to build an object from the "- rest" + subsequent indented lines
                    // For now, handle the simple case: `- key: value` as a one-entry map
                    if let Some((key, val_str)) = split_key_value(rest) {
                        let item_map = TokMap::alloc();
                        self.advance();
                        if val_str.is_empty() {
                            // Check for indented children
                            if let Some(next) = self.peek() {
                                if next.indent > item_indent {
                                    let child = self.parse_object_block(next.indent);
                                    unsafe {
                                        (*item_map).data.insert(key.to_string(), child);
                                    }
                                } else {
                                    unsafe {
                                        (*item_map).data.insert(key.to_string(), TokValue::nil());
                                    }
                                }
                            }
                        } else {
                            let val = parse_primitive(val_str);
                            unsafe {
                                (*item_map).data.insert(key.to_string(), val);
                            }
                        }
                        // Collect any sibling keys at item_indent+2
                        while let Some(next) = self.peek() {
                            if next.indent <= item_indent {
                                break;
                            }
                            if next.content.starts_with("- ") {
                                break;
                            }
                            if let Some((k, v)) = split_key_value(next.content) {
                                let val = if v.is_empty() {
                                    TokValue::nil()
                                } else {
                                    parse_primitive(v)
                                };
                                unsafe {
                                    (*item_map).data.insert(k.to_string(), val);
                                }
                            }
                            self.advance();
                        }
                        unsafe {
                            (*arr).data.push(TokValue::from_map(item_map));
                        }
                    } else {
                        self.advance();
                        unsafe {
                            (*arr).data.push(parse_primitive(rest));
                        }
                    }
                    continue;
                }

                // Check if the item is a key: value pair (starts a map)
                if let Some((key, val_str)) = split_key_value(rest) {
                    let item_map = TokMap::alloc();
                    self.advance();
                    if val_str.is_empty() {
                        // Nested object as array element
                        if let Some(next) = self.peek() {
                            if next.indent > item_indent {
                                let child = self.parse_object_block(next.indent);
                                unsafe {
                                    (*item_map).data.insert(key.to_string(), child);
                                }
                            } else {
                                unsafe {
                                    (*item_map).data.insert(key.to_string(), TokValue::nil());
                                }
                            }
                        }
                    } else {
                        let val = parse_primitive(val_str);
                        unsafe {
                            (*item_map).data.insert(key.to_string(), val);
                        }
                    }
                    // Collect any sibling keys at same depth (item_indent + 2 relative)
                    while let Some(next) = self.peek() {
                        if next.indent <= item_indent {
                            break;
                        }
                        if next.content.starts_with("- ") {
                            break;
                        }
                        if let Some((k, v)) = split_key_value(next.content) {
                            let val = if v.is_empty() {
                                TokValue::nil()
                            } else {
                                parse_primitive(v)
                            };
                            unsafe {
                                (*item_map).data.insert(k.to_string(), val);
                            }
                        }
                        self.advance();
                    }
                    unsafe {
                        (*arr).data.push(TokValue::from_map(item_map));
                    }
                } else {
                    // Simple primitive item
                    self.advance();

                    // Check for inline array header on the "- " content
                    if rest.starts_with('[') {
                        let (_count, delim, fields) = parse_array_header_no_key(rest);
                        if let Some(fields) = fields {
                            let tab_arr = self.parse_tabular_rows(&fields, delim, item_indent);
                            unsafe {
                                (*arr).data.push(tab_arr);
                            }
                        } else {
                            let after = line_after_header_colon(rest);
                            if !after.is_empty() {
                                let prim_arr = parse_csv_primitives(after, delim);
                                unsafe {
                                    (*arr).data.push(prim_arr);
                                }
                            } else {
                                unsafe {
                                    (*arr).data.push(parse_primitive(rest));
                                }
                            }
                        }
                    } else {
                        unsafe {
                            (*arr).data.push(parse_primitive(rest));
                        }
                    }
                }
            } else {
                // Non-hyphenated child — shouldn't happen in mixed array, skip
                self.advance();
            }
        }
        TokValue::from_array(arr)
    }
}

// ═══════════════════════════════════════════════════════════════
// Line-level parsing helpers
// ═══════════════════════════════════════════════════════════════

/// Check if a line looks like a `key: value` pair (colon not inside quotes).
fn line_is_key_value(s: &str) -> bool {
    split_key_value(s).is_some()
}

/// Split `key: value` — returns (key, value_str). The colon must be followed
/// by a space or be at end of string. Colons inside quoted strings are skipped.
fn split_key_value(s: &str) -> Option<(&str, &str)> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut in_quotes = false;
    while i < bytes.len() {
        match bytes[i] {
            b'"' if !in_quotes => in_quotes = true,
            b'"' if in_quotes => in_quotes = false,
            b'\\' if in_quotes => {
                i += 1; // skip escaped char
            }
            b'[' if !in_quotes => {
                // This is an array header, not a simple key-value
                return None;
            }
            b':' if !in_quotes => {
                let key = s[..i].trim();
                if key.is_empty() {
                    return None;
                }
                let rest = if i + 1 < bytes.len() && bytes[i + 1] == b' ' {
                    s[i + 2..].trim_end()
                } else {
                    s[i + 1..].trim()
                };
                return Some((key, rest));
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Try to parse a keyed array header like `key[N]:`, `key[N]: v,v`, `key[N]{f,f}:`.
/// Returns (key, count, delimiter, optional fields).
fn try_parse_keyed_array_header(s: &str) -> Option<(&str, usize, char, Option<Vec<String>>)> {
    let bracket_start = s.find('[')?;
    let key = s[..bracket_start].trim();
    // key can be empty for root arrays, but for keyed headers we require a key
    // Actually for "- [2]: ..." the key would be empty after "- " strip, so allow empty
    let rest = &s[bracket_start..];
    let (count, delim, fields, after) = parse_bracket_header(rest)?;

    // After the header, expect a `:` (possibly with trailing content)
    let after = after.trim_start();
    if !after.starts_with(':') {
        return None;
    }

    Some((key, count, delim, fields))
}

/// Parse `[N]`, `[N|]`, `[N]{f1,f2}`, etc. Returns (count, delim, fields, remaining_str).
fn parse_bracket_header(s: &str) -> Option<(usize, char, Option<Vec<String>>, &str)> {
    if !s.starts_with('[') {
        return None;
    }
    let bracket_end = s.find(']')?;
    let inside = &s[1..bracket_end];

    // Detect delimiter
    let (count_str, delim) = if let Some(stripped) = inside.strip_suffix('|') {
        (stripped, '|')
    } else if let Some(stripped) = inside.strip_suffix('\t') {
        (stripped, '\t')
    } else {
        (inside, ',')
    };

    let count = count_str.parse::<usize>().ok()?;
    let rest = &s[bracket_end + 1..];

    // Check for field list: {f1,f2,...}
    if rest.starts_with('{') {
        let brace_end = rest.find('}')?;
        let fields_str = &rest[1..brace_end];
        let fields: Vec<String> = split_csv(fields_str, delim)
            .into_iter()
            .map(|f| f.to_string())
            .collect();
        let after = &rest[brace_end + 1..];
        Some((count, delim, Some(fields), after))
    } else {
        Some((count, delim, None, rest))
    }
}

/// Parse a root array header (no key prefix): `[N]: ...` or `[N]{f1,f2}:`.
fn parse_array_header_no_key(s: &str) -> (usize, char, Option<Vec<String>>) {
    if let Some((count, delim, fields, _)) = parse_bracket_header(s) {
        (count, delim, fields)
    } else {
        (0, ',', None)
    }
}

/// Get the content after the colon in a header line.
fn line_after_header_colon(s: &str) -> &str {
    // Find the last `:` that's part of the header (after `]` or `}`)
    if let Some(bracket_end) = s.rfind(']') {
        let rest = &s[bracket_end + 1..];
        // Skip optional {fields}
        let rest = if let Some(brace_end) = rest.find('}') {
            &rest[brace_end + 1..]
        } else {
            rest
        };
        if let Some(colon_pos) = rest.find(':') {
            let after = &rest[colon_pos + 1..].trim();
            return after;
        }
    }
    ""
}

/// Split a CSV line by delimiter, respecting quoted strings.
fn split_csv(s: &str, delim: char) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    let bytes = s.as_bytes();
    let delim_byte = delim as u8;

    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' if !in_quotes => in_quotes = true,
            b'"' if in_quotes => in_quotes = false,
            b'\\' if in_quotes => {
                i += 1; // skip escaped char
            }
            b if b == delim_byte && !in_quotes => {
                result.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    result.push(s[start..].trim());
    result
}

/// Parse a CSV of primitive values into a TokArray.
fn parse_csv_primitives(s: &str, delim: char) -> TokValue {
    let arr = TokArray::alloc();
    let parts = split_csv(s, delim);
    unsafe {
        for part in parts {
            if !part.is_empty() {
                (*arr).data.push(parse_primitive(part));
            }
        }
    }
    TokValue::from_array(arr)
}

/// Parse a single primitive value.
fn parse_primitive(s: &str) -> TokValue {
    let s = s.trim();
    match s {
        "" | "null" => TokValue::nil(),
        "true" => TokValue::from_bool(true),
        "false" => TokValue::from_bool(false),
        _ if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 => {
            let inner = &s[1..s.len() - 1];
            let unescaped = unescape_string(inner);
            TokValue::from_string(TokString::alloc(unescaped))
        }
        _ => {
            // Try integer
            if let Ok(i) = s.parse::<i64>() {
                // Guard against leading zeros (except "0" and "-0")
                if !has_leading_zeros(s) {
                    return TokValue::from_int(i);
                }
            }
            // Try float
            if let Ok(f) = s.parse::<f64>() {
                if !has_leading_zeros(s) && s.contains('.') {
                    return TokValue::from_float(f);
                }
            }
            // Unquoted string
            TokValue::from_string(TokString::alloc(s.to_string()))
        }
    }
}

/// Check for leading zeros in a number string (invalid in TOON canonical form).
fn has_leading_zeros(s: &str) -> bool {
    let s = s.strip_prefix('-').unwrap_or(s);
    s.len() > 1 && s.starts_with('0') && !s.starts_with("0.")
}

/// Unescape a TOON string (only 5 valid escape sequences).
fn unescape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(ch);
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════
// TOON Encoder
// ═══════════════════════════════════════════════════════════════

/// Convert a TokValue into a TOON string.
unsafe fn tok_to_toon(tv: &TokValue) -> String {
    let mut out = String::new();
    encode_value(tv, &mut out, 0, true);
    // Remove trailing newline if present
    while out.ends_with('\n') {
        out.pop();
    }
    out
}

/// Encode a TokValue. `is_root` is true for the top-level value.
unsafe fn encode_value(tv: &TokValue, out: &mut String, indent: usize, is_root: bool) {
    match tv.tag {
        TAG_NIL => out.push_str("null"),
        TAG_BOOL => {
            if tv.data.bool_val != 0 {
                out.push_str("true");
            } else {
                out.push_str("false");
            }
        }
        TAG_INT => {
            use std::fmt::Write;
            write!(out, "{}", tv.data.int_val).unwrap();
        }
        TAG_FLOAT => {
            let f = tv.data.float_val;
            if f.is_nan() || f.is_infinite() {
                out.push_str("null");
            } else {
                out.push_str(&format_canonical_float(f));
            }
        }
        TAG_STRING => {
            let p = tv.data.string_ptr;
            if p.is_null() {
                out.push_str("null");
            } else {
                encode_string(&(*p).data, out);
            }
        }
        TAG_ARRAY => {
            let p = tv.data.array_ptr;
            if p.is_null() {
                encode_empty_array(out, is_root);
            } else {
                encode_array(&(*p).data, out, indent, is_root);
            }
        }
        TAG_MAP => {
            let p = tv.data.map_ptr;
            if p.is_null() || (*p).data.is_empty() {
                // Empty object produces no output per spec
            } else {
                encode_map(&(*p).data, out, indent, is_root);
            }
        }
        TAG_TUPLE => {
            let p = tv.data.tuple_ptr;
            if p.is_null() {
                encode_empty_array(out, is_root);
            } else {
                encode_array(&(*p).data, out, indent, is_root);
            }
        }
        TAG_FUNC => out.push_str("null"),
        _ => out.push_str("null"),
    }
}

/// Encode an empty array.
fn encode_empty_array(out: &mut String, _is_root: bool) {
    out.push_str("[0]:");
}

/// Encode a map (object) as indented key-value pairs.
unsafe fn encode_map(
    data: &indexmap::IndexMap<String, TokValue>,
    out: &mut String,
    indent: usize,
    is_root: bool,
) {
    let prefix = " ".repeat(indent);
    for (i, (key, val)) in data.iter().enumerate() {
        if !is_root || i > 0 {
            out.push('\n');
        }
        out.push_str(&prefix);
        encode_key(key, out);

        match val.tag {
            TAG_MAP => {
                let p = val.data.map_ptr;
                if p.is_null() || (*p).data.is_empty() {
                    out.push(':');
                } else {
                    out.push(':');
                    encode_map(&(*p).data, out, indent + 2, false);
                }
            }
            TAG_ARRAY => {
                let p = val.data.array_ptr;
                if p.is_null() {
                    out.push_str("[0]:");
                } else {
                    encode_array_as_field(&(*p).data, out, indent);
                }
            }
            TAG_TUPLE => {
                let p = val.data.tuple_ptr;
                if p.is_null() {
                    out.push_str("[0]:");
                } else {
                    encode_array_as_field(&(*p).data, out, indent);
                }
            }
            _ => {
                out.push_str(": ");
                encode_value(val, out, indent, false);
            }
        }
    }
}

/// Encode an array that is a field value (has a key prefix already written).
unsafe fn encode_array_as_field(data: &[TokValue], out: &mut String, indent: usize) {
    if data.is_empty() {
        out.push_str("[0]:");
        return;
    }

    // Check if this is a uniform array of maps (tabular format)
    if let Some(fields) = get_uniform_map_fields(data) {
        if all_tabular_values_primitive(data, &fields) {
            use std::fmt::Write;
            write!(out, "[{}]{{{}}}:", data.len(), fields.join(",")).unwrap();
            let child_prefix = " ".repeat(indent + 2);
            for item in data {
                out.push('\n');
                out.push_str(&child_prefix);
                let p = item.data.map_ptr;
                for (j, field) in fields.iter().enumerate() {
                    if j > 0 {
                        out.push(',');
                    }
                    if let Some(v) = (*p).data.get(field) {
                        encode_csv_value(v, out);
                    } else {
                        out.push_str("null");
                    }
                }
            }
            return;
        }
    }

    // Check if all elements are primitives (inline format)
    if all_primitives(data) {
        use std::fmt::Write;
        write!(out, "[{}]: ", data.len()).unwrap();
        for (i, v) in data.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            encode_csv_value(v, out);
        }
        return;
    }

    // Mixed array
    use std::fmt::Write;
    write!(out, "[{}]:", data.len()).unwrap();
    let child_prefix = " ".repeat(indent + 2);
    for v in data {
        out.push('\n');
        out.push_str(&child_prefix);
        out.push_str("- ");
        encode_value(v, out, indent + 4, false);
    }
}

/// Encode a root-level array (no key prefix).
unsafe fn encode_array(data: &[TokValue], out: &mut String, indent: usize, is_root: bool) {
    if !is_root {
        // Non-root arrays should not appear standalone (they're under a key)
        // But handle it gracefully
        out.push_str("null");
        return;
    }

    if data.is_empty() {
        out.push_str("[0]:");
        return;
    }

    // Check for tabular
    if let Some(fields) = get_uniform_map_fields(data) {
        if all_tabular_values_primitive(data, &fields) {
            use std::fmt::Write;
            write!(out, "[{}]{{{}}}:", data.len(), fields.join(",")).unwrap();
            let child_prefix = " ".repeat(indent + 2);
            for item in data {
                out.push('\n');
                out.push_str(&child_prefix);
                let p = item.data.map_ptr;
                for (j, field) in fields.iter().enumerate() {
                    if j > 0 {
                        out.push(',');
                    }
                    if let Some(v) = (*p).data.get(field) {
                        encode_csv_value(v, out);
                    } else {
                        out.push_str("null");
                    }
                }
            }
            return;
        }
    }

    // Check for inline primitives
    if all_primitives(data) {
        use std::fmt::Write;
        write!(out, "[{}]: ", data.len()).unwrap();
        for (i, v) in data.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            encode_csv_value(v, out);
        }
        return;
    }

    // Mixed root array
    use std::fmt::Write;
    write!(out, "[{}]:", data.len()).unwrap();
    let child_prefix = " ".repeat(indent + 2);
    for v in data {
        out.push('\n');
        out.push_str(&child_prefix);
        out.push_str("- ");
        encode_value(v, out, indent + 4, false);
    }
}

/// Encode a value for use in a CSV row (no newlines allowed).
unsafe fn encode_csv_value(tv: &TokValue, out: &mut String) {
    match tv.tag {
        TAG_NIL => out.push_str("null"),
        TAG_BOOL => {
            if tv.data.bool_val != 0 {
                out.push_str("true");
            } else {
                out.push_str("false");
            }
        }
        TAG_INT => {
            use std::fmt::Write;
            write!(out, "{}", tv.data.int_val).unwrap();
        }
        TAG_FLOAT => {
            let f = tv.data.float_val;
            if f.is_nan() || f.is_infinite() {
                out.push_str("null");
            } else {
                out.push_str(&format_canonical_float(f));
            }
        }
        TAG_STRING => {
            let p = tv.data.string_ptr;
            if p.is_null() {
                out.push_str("null");
            } else {
                encode_csv_string(&(*p).data, out);
            }
        }
        _ => out.push_str("null"),
    }
}

/// Encode a key name (quote if necessary).
fn encode_key(key: &str, out: &mut String) {
    if key_needs_quoting(key) {
        out.push('"');
        escape_string_content(key, out);
        out.push('"');
    } else {
        out.push_str(key);
    }
}

/// Check if a key needs quoting.
fn key_needs_quoting(s: &str) -> bool {
    s.is_empty()
        || s.contains(':')
        || s.contains('"')
        || s.contains('\\')
        || s.contains('[')
        || s.contains(']')
        || s.contains('{')
        || s.contains('}')
        || s.contains(' ')
        || s.contains('\n')
        || s.contains('\t')
        || s.contains('\r')
}

/// Encode a string value, quoting only if necessary.
fn encode_string(s: &str, out: &mut String) {
    if needs_quoting(s) {
        out.push('"');
        escape_string_content(s, out);
        out.push('"');
    } else {
        out.push_str(s);
    }
}

/// Encode a string for use in a CSV context (must also quote if contains delimiter).
fn encode_csv_string(s: &str, out: &mut String) {
    if needs_quoting(s) || s.contains(',') {
        out.push('"');
        escape_string_content(s, out);
        out.push('"');
    } else {
        out.push_str(s);
    }
}

/// Check if a string value needs quoting.
fn needs_quoting(s: &str) -> bool {
    s.is_empty()
        || s == "true"
        || s == "false"
        || s == "null"
        || looks_like_number(s)
        || s.contains(':')
        || s.contains('"')
        || s.contains('\\')
        || s.contains('[')
        || s.contains(']')
        || s.contains('{')
        || s.contains('}')
        || s.contains(',')
        || s.contains('\n')
        || s.contains('\t')
        || s.contains('\r')
        || s.starts_with(' ')
        || s.ends_with(' ')
        || s.starts_with('-')
}

/// Check if a string looks like a number.
fn looks_like_number(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let s = s.strip_prefix('-').unwrap_or(s);
    if s.is_empty() {
        return false;
    }
    // All digits, or digits.digits
    let mut has_dot = false;
    for ch in s.chars() {
        if ch == '.' && !has_dot {
            has_dot = true;
        } else if !ch.is_ascii_digit() {
            return false;
        }
    }
    true
}

/// Escape string content (without surrounding quotes).
fn escape_string_content(s: &str, out: &mut String) {
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
}

/// Format a float in canonical TOON form (no exponents, no trailing zeros).
fn format_canonical_float(f: f64) -> String {
    if f == 0.0 {
        return "0".to_string();
    }
    // Use enough precision, then strip trailing zeros
    let s = format!("{}", f);
    // Rust's default Display doesn't use exponents for reasonable values
    // but we should handle it just in case
    if s.contains('e') || s.contains('E') {
        // Fall back to a fixed decimal representation
        let s = format!("{:.20}", f);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    } else {
        s
    }
}

/// Check if all array elements are primitive (not maps or arrays).
unsafe fn all_primitives(data: &[TokValue]) -> bool {
    data.iter()
        .all(|v| matches!(v.tag, TAG_NIL | TAG_BOOL | TAG_INT | TAG_FLOAT | TAG_STRING))
}

/// If all elements are maps with identical key sets, return the field names.
unsafe fn get_uniform_map_fields(data: &[TokValue]) -> Option<Vec<String>> {
    if data.is_empty() {
        return None;
    }
    // All must be TAG_MAP
    if !data.iter().all(|v| v.tag == TAG_MAP) {
        return None;
    }
    let first = data[0].data.map_ptr;
    if first.is_null() {
        return None;
    }
    let first_keys: Vec<String> = (*first).data.keys().cloned().collect();
    if first_keys.is_empty() {
        return None;
    }
    for item in &data[1..] {
        let p = item.data.map_ptr;
        if p.is_null() {
            return None;
        }
        let keys: Vec<String> = (*p).data.keys().cloned().collect();
        if keys != first_keys {
            return None;
        }
    }
    Some(first_keys)
}

/// Check if all values in a tabular array's maps are primitive.
unsafe fn all_tabular_values_primitive(data: &[TokValue], fields: &[String]) -> bool {
    for item in data {
        let p = item.data.map_ptr;
        if p.is_null() {
            return false;
        }
        for field in fields {
            if let Some(v) = (*p).data.get(field) {
                if !matches!(v.tag, TAG_NIL | TAG_BOOL | TAG_INT | TAG_FLOAT | TAG_STRING) {
                    return false;
                }
            }
        }
    }
    true
}

// ═══════════════════════════════════════════════════════════════
// Public API: parse / stringify
// ═══════════════════════════════════════════════════════════════

/// Parse a TOON string into a TokValue.
fn toon_parse(input: &str) -> TokValue {
    let mut parser = ToonParser::new(input);
    parser.parse_root()
}

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

/// tparse(toon_string) -> Any
#[no_mangle]
pub extern "C" fn tok_toon_parse_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let s = arg_to_str(tag, data);
        toon_parse(s)
    }
}

/// tstr(value) -> Str
#[no_mangle]
pub extern "C" fn tok_toon_stringify_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let tv = TokValue::from_tag_data(tag, data);
        let s = tok_to_toon(&tv);
        TokValue::from_string(TokString::alloc(s))
    }
}

// ═══════════════════════════════════════════════════════════════
// Module constructor
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_stdlib_toon() -> *mut TokMap {
    let m = TokMap::alloc();

    // Primary names
    insert_func(m, "tparse", tok_toon_parse_t as *const u8, 1);
    insert_func(m, "tstr", tok_toon_stringify_t as *const u8, 1);
    // Legacy names for consistency with json module
    insert_func(m, "parse", tok_toon_parse_t as *const u8, 1);
    insert_func(m, "stringify", tok_toon_stringify_t as *const u8, 1);

    m
}
