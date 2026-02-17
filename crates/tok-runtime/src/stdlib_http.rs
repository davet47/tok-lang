//! Standard library: `@"http"` module.
//!
//! Provides HTTP client functions: hget, hpost, hput, hdel, serve.
//! Uses raw TCP sockets — no external HTTP library dependency.

use crate::closure::TokClosure;
use crate::map::TokMap;
use crate::string::TokString;
use crate::tuple::TokTuple;
use crate::value::{TokValue, TAG_FUNC, TAG_INT, TAG_MAP, TAG_STRING};

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

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

/// Parse a URL into (host, port, path). Only supports http:// for now.
fn parse_url(url: &str) -> Option<(String, u16, String)> {
    let url = url.trim();
    let without_scheme = if url.starts_with("https://") {
        return None; // TLS not supported without external deps
    } else if let Some(rest) = url.strip_prefix("http://") {
        rest
    } else {
        url
    };

    let (host_port, path) = match without_scheme.find('/') {
        Some(idx) => (&without_scheme[..idx], &without_scheme[idx..]),
        None => (without_scheme, "/"),
    };

    let (host, port) = match host_port.find(':') {
        Some(idx) => {
            let port = host_port[idx + 1..].parse::<u16>().unwrap_or(80);
            (&host_port[..idx], port)
        }
        None => (host_port, 80),
    };

    Some((host.to_string(), port, path.to_string()))
}

/// Make an HTTP request and return the response body.
fn http_request(method: &str, url: &str, body: Option<&str>) -> Result<String, String> {
    let (host, port, path) =
        parse_url(url).ok_or_else(|| "Invalid URL (only http:// supported)".to_string())?;

    let mut stream = TcpStream::connect(format!("{}:{}", host, port))
        .map_err(|e| format!("Connection failed: {}", e))?;

    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .ok();

    // Build request
    let body_str = body.unwrap_or("");
    let request = if body_str.is_empty() {
        format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
            method, path, host
        )
    } else {
        format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            method, path, host, body_str.len(), body_str
        )
    };

    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("Write failed: {}", e))?;
    stream.flush().map_err(|e| format!("Flush failed: {}", e))?;

    // Read response
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|e| format!("Read failed: {}", e))?;

    let response_str = String::from_utf8_lossy(&response).to_string();

    // Split headers from body
    if let Some(idx) = response_str.find("\r\n\r\n") {
        let headers = &response_str[..idx];
        let body = &response_str[idx + 4..];

        // Check status code
        if let Some(status_line) = headers.lines().next() {
            let parts: Vec<&str> = status_line.split_whitespace().collect();
            if parts.len() >= 2 {
                let code: u16 = parts[1].parse().unwrap_or(0);
                if code >= 400 {
                    return Err(format!("HTTP {}", code));
                }
            }
        }

        Ok(body.to_string())
    } else {
        Ok(response_str)
    }
}

/// Helper: wrap result as (body, err) tuple
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

// ═══════════════════════════════════════════════════════════════
// Trampolines
// ═══════════════════════════════════════════════════════════════

/// hget(url) -> (body, err)
#[no_mangle]
pub extern "C" fn tok_http_hget_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    let url = unsafe { arg_to_str(tag, data).to_string() };
    to_result_tuple(http_request("GET", &url, None))
}

/// hpost(url, body) -> (body, err)
#[no_mangle]
pub extern "C" fn tok_http_hpost_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let url = arg_to_str(tag1, data1).to_string();
        let body = arg_to_str(tag2, data2).to_string();
        to_result_tuple(http_request("POST", &url, Some(&body)))
    }
}

/// hput(url, body) -> (body, err)
#[no_mangle]
pub extern "C" fn tok_http_hput_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    unsafe {
        let url = arg_to_str(tag1, data1).to_string();
        let body = arg_to_str(tag2, data2).to_string();
        to_result_tuple(http_request("PUT", &url, Some(&body)))
    }
}

/// hdel(url) -> (body, err)
#[no_mangle]
pub extern "C" fn tok_http_hdel_t(_env: *mut u8, tag: i64, data: i64) -> TokValue {
    let url = unsafe { arg_to_str(tag, data).to_string() };
    to_result_tuple(http_request("DELETE", &url, None))
}

/// serve(port, routes) -> Nil
/// Start a basic HTTP server. `routes` is a map of "METHOD /path" -> handler function.
/// Handler functions receive a request map {method, path, body, headers} and return
/// a response string or map {status, body}.
#[no_mangle]
pub extern "C" fn tok_http_serve_t(
    _env: *mut u8,
    tag1: i64,
    data1: i64,
    tag2: i64,
    data2: i64,
) -> TokValue {
    let port = arg_to_i64(tag1, data1);
    if tag2 as u8 != TAG_MAP {
        eprintln!("http.serve error: routes must be a map");
        return TokValue::nil();
    }
    let routes_ptr = data2 as *mut TokMap;
    if routes_ptr.is_null() {
        eprintln!("http.serve error: null routes map");
        return TokValue::nil();
    }

    use std::net::TcpListener;

    let addr = format!("0.0.0.0:{}", port);
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("http.serve error: {}", e);
            return TokValue::nil();
        }
    };

    eprintln!("Tok HTTP server listening on {}", addr);

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("http.serve connection error: {}", e);
                continue;
            }
        };

        let mut reader = BufReader::new(&stream);
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() {
            continue;
        }

        // Parse request line: "GET /path HTTP/1.1"
        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let method = parts[0];
        let path = parts[1];

        // Read headers
        let mut content_length: usize = 0;
        let headers_map = TokMap::alloc();
        loop {
            let mut header_line = String::new();
            if reader.read_line(&mut header_line).is_err() {
                break;
            }
            let trimmed = header_line.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some((key, val)) = trimmed.split_once(':') {
                let key = key.trim().to_lowercase();
                let val = val.trim().to_string();
                if key == "content-length" {
                    content_length = val.parse().unwrap_or(0);
                }
                unsafe {
                    (*headers_map)
                        .data
                        .insert(key, TokValue::from_string(TokString::alloc(val)));
                }
            }
        }

        // Read body
        let body = if content_length > 0 {
            let mut body_buf = vec![0u8; content_length];
            use std::io::Read;
            if reader.read_exact(&mut body_buf).is_ok() {
                String::from_utf8_lossy(&body_buf).to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Build request map
        let req_map = TokMap::alloc();
        unsafe {
            (*req_map).data.insert(
                "method".to_string(),
                TokValue::from_string(TokString::alloc(method.to_string())),
            );
            (*req_map).data.insert(
                "path".to_string(),
                TokValue::from_string(TokString::alloc(path.to_string())),
            );
            (*req_map).data.insert(
                "body".to_string(),
                TokValue::from_string(TokString::alloc(body)),
            );
            (*req_map)
                .data
                .insert("headers".to_string(), TokValue::from_map(headers_map));
        }

        // Look up route handler
        let route_key = format!("{} {}", method, path);
        let handler = unsafe {
            (*routes_ptr)
                .data
                .get(&route_key)
                .or_else(|| (*routes_ptr).data.get(&format!("* {}", path)))
                .or_else(|| (*routes_ptr).data.get("*"))
                .copied()
        };

        let (status, response_body) = if let Some(handler_val) = handler {
            if handler_val.tag == TAG_FUNC {
                let closure_ptr = unsafe { handler_val.data.func_ptr };
                if !closure_ptr.is_null() {
                    let fn_ptr = unsafe { (*closure_ptr).fn_ptr };
                    let env_ptr = unsafe { (*closure_ptr).env_ptr };
                    // Call handler with request map as TokValue arg
                    let req_val = TokValue::from_map(req_map);
                    let call: extern "C" fn(*mut u8, i64, i64) -> TokValue =
                        unsafe { std::mem::transmute(fn_ptr) };
                    let result = call(env_ptr, req_val.tag as i64, unsafe {
                        req_val.data._raw as i64
                    });
                    // Parse result: string → body, map → {status, body}
                    match result.tag {
                        TAG_STRING => {
                            let body_str = if unsafe { result.data.string_ptr.is_null() } {
                                String::new()
                            } else {
                                unsafe { (*result.data.string_ptr).data.clone() }
                            };
                            (200, body_str)
                        }
                        TAG_MAP => {
                            let resp_map = unsafe { result.data.map_ptr };
                            if resp_map.is_null() {
                                (200, String::new())
                            } else {
                                let status = unsafe {
                                    (*resp_map)
                                        .data
                                        .get("status")
                                        .map(|v| {
                                            if v.tag == TAG_INT {
                                                v.data.int_val as u16
                                            } else {
                                                200
                                            }
                                        })
                                        .unwrap_or(200)
                                };
                                let body_str = unsafe {
                                    (*resp_map)
                                        .data
                                        .get("body")
                                        .map(|v| {
                                            if v.tag == TAG_STRING && !v.data.string_ptr.is_null() {
                                                (*v.data.string_ptr).data.clone()
                                            } else {
                                                format!("{}", v)
                                            }
                                        })
                                        .unwrap_or_default()
                                };
                                (status, body_str)
                            }
                        }
                        _ => (200, format!("{}", result)),
                    }
                } else {
                    (500, "Handler function pointer is null".to_string())
                }
            } else {
                (500, "Route handler is not a function".to_string())
            }
        } else {
            (404, "Not Found".to_string())
        };

        // Write HTTP response
        let response = format!(
            "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            status,
            match status {
                200 => "OK",
                404 => "Not Found",
                500 => "Internal Server Error",
                _ => "Unknown",
            },
            response_body.len(),
            response_body
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
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
pub extern "C" fn tok_stdlib_http() -> *mut TokMap {
    let m = TokMap::alloc();

    insert_func(m, "hget", tok_http_hget_t as *const u8, 1);
    insert_func(m, "hpost", tok_http_hpost_t as *const u8, 2);
    insert_func(m, "hput", tok_http_hput_t as *const u8, 2);
    insert_func(m, "hdel", tok_http_hdel_t as *const u8, 1);
    insert_func(m, "serve", tok_http_serve_t as *const u8, 2);

    m
}
