//! Standard library: `@"llm"` module.
//!
//! Provides LLM API calls with automatic provider detection.
//! - `ask(prompt)` — one-shot completion → `(response, error)`
//! - `chat(messages, opts)` — multi-turn with options → `(response, error)`
//!
//! Supports OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) natively.
//! Provider auto-detected from env vars: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

use crate::map::TokMap;
use crate::stdlib_http::http_request_with_headers;
use crate::string::TokString;
use crate::tuple::TokTuple;
use crate::value::{TokValue, TAG_ARRAY, TAG_FLOAT, TAG_INT, TAG_MAP, TAG_NIL, TAG_STRING};

use crate::stdlib_helpers::{arg_to_str, insert_func};

/// Get a string field from a TokMap, or default.
unsafe fn map_get_str(map: *const TokMap, key: &str) -> Option<String> {
    if map.is_null() {
        return None;
    }
    (*map).data.get(key).and_then(|v| {
        if v.tag == TAG_STRING && !v.data.string_ptr.is_null() {
            Some((*v.data.string_ptr).data.clone())
        } else {
            None
        }
    })
}

/// Get an int field from a TokMap.
unsafe fn map_get_int(map: *const TokMap, key: &str) -> Option<i64> {
    if map.is_null() {
        return None;
    }
    (*map).data.get(key).and_then(|v| {
        if v.tag == TAG_INT {
            Some(v.data.int_val)
        } else {
            None
        }
    })
}

/// Get a float field from a TokMap.
unsafe fn map_get_float(map: *const TokMap, key: &str) -> Option<f64> {
    if map.is_null() {
        return None;
    }
    (*map).data.get(key).and_then(|v| {
        if v.tag == TAG_FLOAT {
            Some(v.data.float_val)
        } else if v.tag == TAG_INT {
            Some(v.data.int_val as f64)
        } else {
            None
        }
    })
}

fn to_result_tuple(result: Result<String, String>) -> TokValue {
    match result {
        Ok(body) => {
            let elems = vec![
                TokValue::from_string(TokString::alloc(body)),
                TokValue::nil(),
            ];
            TokValue::from_tuple(TokTuple::alloc(elems))
        }
        Err(err) => {
            let elems = vec![
                TokValue::nil(),
                TokValue::from_string(TokString::alloc(err)),
            ];
            TokValue::from_tuple(TokTuple::alloc(elems))
        }
    }
}

/// Escape a string for JSON (handle backslash, quotes, newlines, tabs).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════
// Provider Configuration
// ═══════════════════════════════════════════════════════════════

enum Provider {
    OpenAI,
    Anthropic,
}

struct ProviderConfig {
    provider: Provider,
    url: String,
    api_key: String,
    model: String,
    max_tokens: Option<i64>,
    temperature: Option<f64>,
    system: Option<String>,
}

/// Detect provider from opts map and environment variables.
unsafe fn detect_provider(opts: *const TokMap) -> Result<ProviderConfig, String> {
    let explicit_provider = map_get_str(opts, "provider");
    let explicit_url = map_get_str(opts, "url");
    let explicit_key = map_get_str(opts, "api_key");
    let explicit_model = map_get_str(opts, "model");
    let max_tokens = map_get_int(opts, "max_tokens");
    let temperature = map_get_float(opts, "temperature");
    let system = map_get_str(opts, "system");

    // Custom URL → OpenAI-compatible format
    if let Some(url) = explicit_url {
        let api_key = explicit_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .unwrap_or_default();
        return Ok(ProviderConfig {
            provider: Provider::OpenAI,
            url,
            api_key,
            model: explicit_model.unwrap_or_else(|| "default".to_string()),
            max_tokens,
            temperature,
            system,
        });
    }

    // Explicit provider
    if let Some(ref p) = explicit_provider {
        match p.as_str() {
            "anthropic" => {
                let api_key = explicit_key
                    .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                    .ok_or("ANTHROPIC_API_KEY not set")?;
                return Ok(ProviderConfig {
                    provider: Provider::Anthropic,
                    url: "https://api.anthropic.com/v1/messages".to_string(),
                    api_key,
                    model: explicit_model.unwrap_or_else(|| "claude-sonnet-4-20250514".to_string()),
                    max_tokens,
                    temperature,
                    system,
                });
            }
            "openai" => {
                let api_key = explicit_key
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    .ok_or("OPENAI_API_KEY not set")?;
                return Ok(ProviderConfig {
                    provider: Provider::OpenAI,
                    url: "https://api.openai.com/v1/chat/completions".to_string(),
                    api_key,
                    model: explicit_model.unwrap_or_else(|| "gpt-4o".to_string()),
                    max_tokens,
                    temperature,
                    system,
                });
            }
            other => return Err(format!("Unknown provider: {}", other)),
        }
    }

    // Auto-detect from env vars (Anthropic first)
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        return Ok(ProviderConfig {
            provider: Provider::Anthropic,
            url: "https://api.anthropic.com/v1/messages".to_string(),
            api_key: key,
            model: explicit_model.unwrap_or_else(|| "claude-sonnet-4-20250514".to_string()),
            max_tokens,
            temperature,
            system,
        });
    }

    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        return Ok(ProviderConfig {
            provider: Provider::OpenAI,
            url: "https://api.openai.com/v1/chat/completions".to_string(),
            api_key: key,
            model: explicit_model.unwrap_or_else(|| "gpt-4o".to_string()),
            max_tokens,
            temperature,
            system,
        });
    }

    Err("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY".to_string())
}

// ═══════════════════════════════════════════════════════════════
// Request / Response Builders
// ═══════════════════════════════════════════════════════════════

/// Convert tok messages array to JSON messages string: `[{"role":"...","content":"..."},...]`
unsafe fn messages_to_json(messages: *const crate::array::TokArray) -> String {
    if messages.is_null() {
        return "[]".to_string();
    }
    let mut parts = Vec::new();
    for msg in &(*messages).data {
        if msg.tag != TAG_MAP {
            continue;
        }
        let m = msg.data.map_ptr;
        if m.is_null() {
            continue;
        }
        let role = map_get_str(m, "role").unwrap_or_else(|| "user".to_string());
        let content = map_get_str(m, "content").unwrap_or_default();
        parts.push(format!(
            "{{\"role\":\"{}\",\"content\":\"{}\"}}",
            json_escape(&role),
            json_escape(&content)
        ));
    }
    format!("[{}]", parts.join(","))
}

/// Build OpenAI-compatible request body.
unsafe fn build_openai_body(
    messages: *const crate::array::TokArray,
    config: &ProviderConfig,
) -> String {
    let mut msgs_json = String::new();

    // Prepend system message if provided
    if let Some(ref sys) = config.system {
        msgs_json.push_str(&format!(
            "{{\"role\":\"system\",\"content\":\"{}\"}},",
            json_escape(sys)
        ));
    }

    let user_msgs = messages_to_json(messages);
    // Strip the surrounding [] from user messages and append
    let inner = &user_msgs[1..user_msgs.len() - 1];
    msgs_json.push_str(inner);

    let mut body = format!(
        "{{\"model\":\"{}\",\"messages\":[{}]",
        json_escape(&config.model),
        msgs_json
    );

    if let Some(mt) = config.max_tokens {
        body.push_str(&format!(",\"max_tokens\":{}", mt));
    }
    if let Some(temp) = config.temperature {
        body.push_str(&format!(",\"temperature\":{}", temp));
    }
    body.push('}');
    body
}

/// Build Anthropic request body.
unsafe fn build_anthropic_body(
    messages: *const crate::array::TokArray,
    config: &ProviderConfig,
) -> String {
    let max_tokens = config.max_tokens.unwrap_or(4096);

    let mut body = format!(
        "{{\"model\":\"{}\",\"max_tokens\":{}",
        json_escape(&config.model),
        max_tokens
    );

    if let Some(ref sys) = config.system {
        body.push_str(&format!(",\"system\":\"{}\"", json_escape(sys)));
    }
    if let Some(temp) = config.temperature {
        body.push_str(&format!(",\"temperature\":{}", temp));
    }

    let msgs_json = messages_to_json(messages);
    body.push_str(&format!(",\"messages\":{}", msgs_json));
    body.push('}');
    body
}

/// Parse OpenAI response JSON → extract content string.
fn parse_openai_response(body: &str) -> Result<String, String> {
    let json: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("JSON parse error: {}", e))?;

    // Check for error response
    if let Some(err) = json.get("error") {
        let msg = err
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        return Err(format!("API error: {}", msg));
    }

    json.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("Unexpected response format: {}", body))
}

/// Parse Anthropic response JSON → extract content string.
fn parse_anthropic_response(body: &str) -> Result<String, String> {
    let json: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("JSON parse error: {}", e))?;

    // Check for error response
    if let Some(err) = json.get("error") {
        let msg = err
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        return Err(format!("API error: {}", msg));
    }

    json.get("content")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("Unexpected response format: {}", body))
}

// ═══════════════════════════════════════════════════════════════
// Core chat function
// ═══════════════════════════════════════════════════════════════

unsafe fn llm_chat(
    messages: *const crate::array::TokArray,
    opts: *const TokMap,
) -> Result<String, String> {
    let config = detect_provider(opts)?;
    let is_anthropic = matches!(config.provider, Provider::Anthropic);

    let body = if is_anthropic {
        build_anthropic_body(messages, &config)
    } else {
        build_openai_body(messages, &config)
    };

    let headers: Vec<(&str, String)> = if is_anthropic {
        vec![
            ("x-api-key", config.api_key.clone()),
            ("anthropic-version", "2023-06-01".to_string()),
        ]
    } else {
        let mut hdrs = vec![];
        if !config.api_key.is_empty() {
            hdrs.push(("Authorization", format!("Bearer {}", config.api_key)));
        }
        hdrs
    };

    let header_refs: Vec<(&str, &str)> = headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let response = http_request_with_headers("POST", &config.url, Some(&body), &header_refs, 120)?;

    if is_anthropic {
        parse_anthropic_response(&response)
    } else {
        parse_openai_response(&response)
    }
}

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

/// ask(prompt) -> (response, error)
#[no_mangle]
pub extern "C" fn tok_llm_ask_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    unsafe {
        let prompt = arg_to_str(tag, data);
        if prompt.is_empty() {
            return to_result_tuple(Err("ask: prompt cannot be empty".to_string()));
        }

        // Build a single-message array: [{role:"user", content:prompt}]
        let msg_map = TokMap::alloc();
        (*msg_map).data.insert(
            "role".to_string(),
            TokValue::from_string(TokString::alloc("user".to_string())),
        );
        (*msg_map).data.insert(
            "content".to_string(),
            TokValue::from_string(TokString::alloc(prompt.to_string())),
        );
        let arr = crate::array::TokArray::alloc();
        (*arr).data.push(TokValue::from_map(msg_map));

        // Empty opts
        let empty_opts = TokMap::alloc();
        let result = llm_chat(arr, empty_opts);

        // Free temporaries: msg_map is nested inside arr, so rc_dec arr first
        // (which will rc_dec the map element), then free empty_opts.
        TokValue::from_array(arr).rc_dec();
        TokValue::from_map(empty_opts).rc_dec();

        to_result_tuple(result)
    }
}

/// chat(messages, opts) -> (response, error)
#[no_mangle]
pub extern "C" fn tok_llm_chat_2_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        // First arg: messages array
        let messages = if tag1 as u8 == TAG_ARRAY {
            data1 as *const crate::array::TokArray
        } else if tag1 as u8 == TAG_NIL {
            return to_result_tuple(Err("chat: messages cannot be nil".to_string()));
        } else {
            return to_result_tuple(Err("chat: first argument must be an array".to_string()));
        };

        if messages.is_null() || (*messages).data.is_empty() {
            return to_result_tuple(Err("chat: messages array is empty".to_string()));
        }

        // Second arg: opts map (may be nil or empty)
        let opts = if tag2 as u8 == TAG_MAP {
            data2 as *const TokMap
        } else {
            std::ptr::null()
        };

        let result = llm_chat(messages, opts);
        to_result_tuple(result)
    }
}

// ═══════════════════════════════════════════════════════════════
// Module constructor
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn tok_stdlib_llm() -> *mut TokMap {
    let m = TokMap::alloc();

    insert_func(m, "ask", tok_llm_ask_t as *const u8, 1);
    insert_func(m, "chat", tok_llm_chat_2_t as *const u8, 2);

    m
}
