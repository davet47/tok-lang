//! Standard library: `@"tmpl"` module.
//!
//! Provides template rendering with `{.key.}` syntax.
//! - `render(template, data)` — parse and render in one step
//! - `compile(template)` — parse template → compiled AST (reusable)
//! - `apply(compiled, data)` — render a compiled template with data

use crate::array::TokArray;
use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{
    TokValue, TAG_ARRAY, TAG_BOOL, TAG_FLOAT, TAG_INT, TAG_MAP, TAG_NIL, TAG_STRING,
};

use crate::stdlib_helpers::{arg_to_str, insert_func};

const MAX_DEPTH: usize = 128;

/// Convert a TokValue to its string representation for template output.
unsafe fn value_to_string(tv: &TokValue) -> String {
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
        TAG_FLOAT => {
            let f = tv.data.float_val;
            let s = format!("{}", f);
            s
        }
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

// ═══════════════════════════════════════════════════════════════
// Template AST
// ═══════════════════════════════════════════════════════════════

/// A parsed template node.
enum Node {
    Text(String),
    Var(String),                 // key (may contain dots)
    SelfRef,                     // {. . .} = current element
    Section(String, Vec<Node>),  // key, children
    Inverted(String, Vec<Node>), // key, children
}

// ═══════════════════════════════════════════════════════════════
// Template Parser
// ═══════════════════════════════════════════════════════════════

/// Parse a template string into a list of AST nodes.
fn parse_template(input: &str) -> Vec<Node> {
    let mut nodes = Vec::new();
    let mut pos = 0;
    let bytes = input.as_bytes();
    let len = bytes.len();

    while pos < len {
        // Look for `{.`
        if let Some(tag_start) = find_open_tag(bytes, pos) {
            // Emit any text before the tag
            if tag_start > pos {
                nodes.push(Node::Text(input[pos..tag_start].to_string()));
            }

            // Find matching `.}`
            if let Some(tag_end) = find_close_tag(bytes, tag_start + 2) {
                let inner = &input[tag_start + 2..tag_end];
                let inner = inner.trim();

                if inner == ".." {
                    // {. .. .} = self reference (the 3-dot form is {.  ..  .} with dots inside)
                    // Actually {. . .} has inner = ".", and {... } is special
                    nodes.push(Node::SelfRef);
                } else if let Some(rest) = inner.strip_prefix('#') {
                    // Section open
                    let key = rest.trim().to_string();
                    // Parse children until matching {./key.}
                    let close_tag = format!("/{}", key);
                    let (children, new_pos) = parse_section_body(input, tag_end + 2, &close_tag, 0);
                    nodes.push(Node::Section(key, children));
                    pos = new_pos;
                    continue;
                } else if let Some(rest) = inner.strip_prefix('^') {
                    // Inverted section open
                    let key = rest.trim().to_string();
                    let close_tag = format!("/{}", key);
                    let (children, new_pos) = parse_section_body(input, tag_end + 2, &close_tag, 0);
                    nodes.push(Node::Inverted(key, children));
                    pos = new_pos;
                    continue;
                } else if inner.starts_with('!') {
                    // Comment — skip
                } else if inner.starts_with('/') {
                    // Closing tag at wrong level — skip (shouldn't happen in well-formed templates)
                } else {
                    // Variable
                    nodes.push(Node::Var(inner.to_string()));
                }
                pos = tag_end + 2;
            } else {
                // No closing .} — emit rest as text
                nodes.push(Node::Text(input[pos..].to_string()));
                pos = len;
            }
        } else {
            // Check for {. . .} literal pattern: three dots
            if pos + 5 <= len && &input[pos..pos + 5] == "{...}" {
                nodes.push(Node::SelfRef);
                pos += 5;
                continue;
            }
            // No more tags — emit remaining text
            nodes.push(Node::Text(input[pos..].to_string()));
            pos = len;
        }
    }

    nodes
}

/// Find `{.` starting from `pos`.
fn find_open_tag(bytes: &[u8], start: usize) -> Option<usize> {
    let len = bytes.len();
    let mut i = start;
    while i + 1 < len {
        // Check for {. . .} shorthand first
        if i + 4 < len
            && bytes[i] == b'{'
            && bytes[i + 1] == b'.'
            && bytes[i + 2] == b'.'
            && bytes[i + 3] == b'.'
            && bytes[i + 4] == b'}'
        {
            return Some(i);
        }
        if bytes[i] == b'{' && bytes[i + 1] == b'.' {
            // Make sure it's not the {. . .} shorthand (already checked above)
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Find `.}` starting from `pos`.
fn find_close_tag(bytes: &[u8], start: usize) -> Option<usize> {
    let len = bytes.len();
    let mut i = start;
    while i + 1 < len {
        if bytes[i] == b'.' && bytes[i + 1] == b'}' {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Parse template body until we find the matching close tag `{./key.}`.
/// Returns (children, position after the close tag).
fn parse_section_body(
    input: &str,
    start: usize,
    close_tag: &str,
    depth: usize,
) -> (Vec<Node>, usize) {
    if depth >= MAX_DEPTH {
        return (Vec::new(), input.len());
    }
    let mut nodes = Vec::new();
    let mut pos = start;
    let bytes = input.as_bytes();
    let len = bytes.len();

    while pos < len {
        // Check for {. . .} shorthand
        if pos + 4 < len
            && bytes[pos] == b'{'
            && bytes[pos + 1] == b'.'
            && bytes[pos + 2] == b'.'
            && bytes[pos + 3] == b'.'
            && bytes[pos + 4] == b'}'
        {
            nodes.push(Node::SelfRef);
            pos += 5;
            continue;
        }

        if let Some(tag_start) = find_open_tag(bytes, pos) {
            // Check if it's the {. . .} form at this position
            if tag_start == pos
                && pos + 4 < len
                && bytes[pos + 1] == b'.'
                && bytes[pos + 2] == b'.'
                && bytes[pos + 3] == b'.'
                && bytes[pos + 4] == b'}'
            {
                nodes.push(Node::SelfRef);
                pos += 5;
                continue;
            }

            // Emit text before tag
            if tag_start > pos {
                nodes.push(Node::Text(input[pos..tag_start].to_string()));
            }

            if let Some(tag_end) = find_close_tag(bytes, tag_start + 2) {
                let inner = &input[tag_start + 2..tag_end];
                let inner = inner.trim();

                // Check if this is our closing tag
                if inner == close_tag {
                    return (nodes, tag_end + 2);
                }

                if inner == ".." {
                    nodes.push(Node::SelfRef);
                } else if let Some(rest) = inner.strip_prefix('#') {
                    let key = rest.trim().to_string();
                    let nested_close = format!("/{}", key);
                    let (children, new_pos) =
                        parse_section_body(input, tag_end + 2, &nested_close, depth + 1);
                    nodes.push(Node::Section(key, children));
                    pos = new_pos;
                    continue;
                } else if let Some(rest) = inner.strip_prefix('^') {
                    let key = rest.trim().to_string();
                    let nested_close = format!("/{}", key);
                    let (children, new_pos) =
                        parse_section_body(input, tag_end + 2, &nested_close, depth + 1);
                    nodes.push(Node::Inverted(key, children));
                    pos = new_pos;
                    continue;
                } else if inner.starts_with('!') {
                    // Comment — skip
                } else if inner.starts_with('/') {
                    // Unexpected close tag — skip
                } else {
                    nodes.push(Node::Var(inner.to_string()));
                }
                pos = tag_end + 2;
            } else {
                nodes.push(Node::Text(input[pos..].to_string()));
                pos = len;
            }
        } else {
            nodes.push(Node::Text(input[pos..].to_string()));
            pos = len;
        }
    }

    (nodes, pos)
}

// ═══════════════════════════════════════════════════════════════
// Template Renderer
// ═══════════════════════════════════════════════════════════════

/// Render a list of AST nodes with a context stack.
unsafe fn render_nodes(nodes: &[Node], ctx: &mut Vec<TokValue>, out: &mut String) {
    for node in nodes {
        match node {
            Node::Text(text) => out.push_str(text),
            Node::Var(key) => {
                if let Some(val) = resolve_key(key, ctx) {
                    out.push_str(&value_to_string(&val));
                }
            }
            Node::SelfRef => {
                if let Some(current) = ctx.last() {
                    out.push_str(&value_to_string(current));
                }
            }
            Node::Section(key, children) => {
                if let Some(val) = resolve_key(key, ctx) {
                    match val.tag {
                        TAG_ARRAY => {
                            let arr_ptr = val.data.array_ptr;
                            if !arr_ptr.is_null() {
                                for item in &(*arr_ptr).data {
                                    ctx.push(*item);
                                    render_nodes(children, ctx, out);
                                    ctx.pop();
                                }
                            }
                        }
                        TAG_MAP => {
                            // Truthy map — push as context
                            ctx.push(val);
                            render_nodes(children, ctx, out);
                            ctx.pop();
                        }
                        TAG_BOOL => {
                            if val.data.bool_val != 0 {
                                render_nodes(children, ctx, out);
                            }
                        }
                        TAG_NIL => {
                            // Falsy — skip
                        }
                        TAG_INT => {
                            if val.data.int_val != 0 {
                                render_nodes(children, ctx, out);
                            }
                        }
                        TAG_STRING => {
                            let ptr = val.data.string_ptr;
                            if !ptr.is_null() && !(*ptr).data.is_empty() {
                                render_nodes(children, ctx, out);
                            }
                        }
                        _ => {
                            // Other truthy values
                            render_nodes(children, ctx, out);
                        }
                    }
                }
                // Missing key = falsy, skip section
            }
            Node::Inverted(key, children) => {
                let is_falsy = if let Some(val) = resolve_key(key, ctx) {
                    match val.tag {
                        TAG_NIL => true,
                        TAG_BOOL => val.data.bool_val == 0,
                        TAG_INT => val.data.int_val == 0,
                        TAG_ARRAY => {
                            let arr_ptr = val.data.array_ptr;
                            arr_ptr.is_null() || (*arr_ptr).data.is_empty()
                        }
                        TAG_STRING => {
                            let ptr = val.data.string_ptr;
                            ptr.is_null() || (*ptr).data.is_empty()
                        }
                        _ => false,
                    }
                } else {
                    true // Missing key = falsy
                };
                if is_falsy {
                    render_nodes(children, ctx, out);
                }
            }
        }
    }
}

/// Resolve a key against the context stack.
/// Supports dot notation: "user.name" resolves to ctx[top]["user"]["name"].
/// Walks from top of stack to bottom to find a match.
unsafe fn resolve_key(key: &str, ctx: &[TokValue]) -> Option<TokValue> {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.is_empty() {
        return None;
    }

    // Walk context stack from top to bottom
    for ctx_val in ctx.iter().rev() {
        if let Some(result) = resolve_parts(&parts, ctx_val) {
            return Some(result);
        }
    }
    None
}

/// Resolve a chain of key parts against a value.
unsafe fn resolve_parts(parts: &[&str], val: &TokValue) -> Option<TokValue> {
    if parts.is_empty() {
        return Some(*val);
    }
    if val.tag != TAG_MAP {
        return None;
    }
    let map_ptr = val.data.map_ptr;
    if map_ptr.is_null() {
        return None;
    }
    let first = parts[0];
    if let Some(child) = (*map_ptr).data.get(first) {
        if parts.len() == 1 {
            Some(*child)
        } else {
            resolve_parts(&parts[1..], child)
        }
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════
// AST Serialization (for compile/apply)
// ═══════════════════════════════════════════════════════════════

/// Serialize AST nodes to TokValue (TokArray of TokMaps).
fn nodes_to_tokvalue(nodes: &[Node]) -> TokValue {
    let arr = TokArray::alloc();
    for node in nodes {
        let map = node_to_tokmap(node);
        unsafe {
            (*arr).data.push(TokValue::from_map(map));
        }
    }
    TokValue::from_array(arr)
}

fn node_to_tokmap(node: &Node) -> *mut TokMap {
    let m = TokMap::alloc();
    match node {
        Node::Text(text) => unsafe {
            (*m).data.insert("type".to_string(), tok_str_val("text"));
            (*m).data.insert("value".to_string(), tok_str_val(text));
        },
        Node::Var(key) => unsafe {
            (*m).data.insert("type".to_string(), tok_str_val("var"));
            (*m).data.insert("key".to_string(), tok_str_val(key));
        },
        Node::SelfRef => unsafe {
            (*m).data.insert("type".to_string(), tok_str_val("self"));
        },
        Node::Section(key, children) => unsafe {
            (*m).data.insert("type".to_string(), tok_str_val("section"));
            (*m).data.insert("key".to_string(), tok_str_val(key));
            (*m).data
                .insert("body".to_string(), nodes_to_tokvalue(children));
        },
        Node::Inverted(key, children) => unsafe {
            (*m).data
                .insert("type".to_string(), tok_str_val("inverted"));
            (*m).data.insert("key".to_string(), tok_str_val(key));
            (*m).data
                .insert("body".to_string(), nodes_to_tokvalue(children));
        },
    }
    m
}

fn tok_str_val(s: &str) -> TokValue {
    TokValue::from_string(TokString::alloc(s.to_string()))
}

/// Deserialize TokValue AST back into Node vec.
unsafe fn tokvalue_to_nodes(tv: &TokValue) -> Vec<Node> {
    if tv.tag != TAG_ARRAY {
        return Vec::new();
    }
    let arr_ptr = tv.data.array_ptr;
    if arr_ptr.is_null() {
        return Vec::new();
    }
    let mut nodes = Vec::new();
    for item in &(*arr_ptr).data {
        if item.tag != TAG_MAP {
            continue;
        }
        let map_ptr = item.data.map_ptr;
        if map_ptr.is_null() {
            continue;
        }
        let map = &(*map_ptr).data;
        let type_str = map
            .get("type")
            .map(|v| value_to_string(v))
            .unwrap_or_default();
        match type_str.as_str() {
            "text" => {
                let value = map
                    .get("value")
                    .map(|v| value_to_string(v))
                    .unwrap_or_default();
                nodes.push(Node::Text(value));
            }
            "var" => {
                let key = map
                    .get("key")
                    .map(|v| value_to_string(v))
                    .unwrap_or_default();
                nodes.push(Node::Var(key));
            }
            "self" => {
                nodes.push(Node::SelfRef);
            }
            "section" => {
                let key = map
                    .get("key")
                    .map(|v| value_to_string(v))
                    .unwrap_or_default();
                let children = map
                    .get("body")
                    .map(|v| tokvalue_to_nodes(v))
                    .unwrap_or_default();
                nodes.push(Node::Section(key, children));
            }
            "inverted" => {
                let key = map
                    .get("key")
                    .map(|v| value_to_string(v))
                    .unwrap_or_default();
                let children = map
                    .get("body")
                    .map(|v| tokvalue_to_nodes(v))
                    .unwrap_or_default();
                nodes.push(Node::Inverted(key, children));
            }
            _ => {}
        }
    }
    nodes
}

// ═══════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════

/// Parse template + render with data.
unsafe fn tmpl_render(template: &str, data: &TokValue) -> String {
    let nodes = parse_template(template);
    let mut ctx = vec![*data];
    let mut out = String::new();
    render_nodes(&nodes, &mut ctx, &mut out);
    out
}

/// Parse template → compiled AST map.
fn tmpl_compile(template: &str) -> TokValue {
    let nodes = parse_template(template);
    let compiled = TokMap::alloc();
    unsafe {
        (*compiled)
            .data
            .insert("nodes".to_string(), nodes_to_tokvalue(&nodes));
    }
    TokValue::from_map(compiled)
}

/// Render compiled AST with data.
unsafe fn tmpl_apply(compiled: &TokValue, data: &TokValue) -> String {
    if compiled.tag != TAG_MAP {
        return String::new();
    }
    let map_ptr = compiled.data.map_ptr;
    if map_ptr.is_null() {
        return String::new();
    }
    let nodes_val = match (*map_ptr).data.get("nodes") {
        Some(v) => v,
        None => return String::new(),
    };
    let nodes = tokvalue_to_nodes(nodes_val);
    let mut ctx = vec![*data];
    let mut out = String::new();
    render_nodes(&nodes, &mut ctx, &mut out);
    out
}

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

/// render(template_string, data_map) -> Str
#[no_mangle]
pub extern "C" fn tok_tmpl_render_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let template = arg_to_str(tag1, data1);
        let data = TokValue::from_tag_data(tag2, data2);
        let result = tmpl_render(template, &data);
        TokValue::from_string(TokString::alloc(result))
    }
}

/// compile(template_string) -> Map (compiled AST)
#[no_mangle]
pub extern "C" fn tok_tmpl_compile_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let template = arg_to_str(tag, data);
        tmpl_compile(template)
    }
}

/// apply(compiled_map, data_map) -> Str
#[no_mangle]
pub extern "C" fn tok_tmpl_apply_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let compiled = TokValue::from_tag_data(tag1, data1);
        let data = TokValue::from_tag_data(tag2, data2);
        let result = tmpl_apply(&compiled, &data);
        TokValue::from_string(TokString::alloc(result))
    }
}

// ═══════════════════════════════════════════════════════════════
// Module constructor
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_stdlib_tmpl() -> *mut TokMap {
    let m = TokMap::alloc();

    insert_func(m, "render", tok_tmpl_render_t as *const u8, 2);
    insert_func(m, "compile", tok_tmpl_compile_t as *const u8, 1);
    insert_func(m, "apply", tok_tmpl_apply_t as *const u8, 2);

    m
}
