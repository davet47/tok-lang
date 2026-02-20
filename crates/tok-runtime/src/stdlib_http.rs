//! Standard library: `@"http"` module.
//!
//! Provides HTTP client functions: hget, hpost, hput, hdel, serve.
//! Supports both HTTP and HTTPS (via rustls).

use crate::map::TokMap;
use crate::string::TokString;
use crate::value::{TokValue, TAG_FUNC, TAG_INT, TAG_MAP, TAG_STRING};

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, OnceLock};

static TLS_CONFIG: OnceLock<Arc<rustls::ClientConfig>> = OnceLock::new();

fn get_tls_config() -> Arc<rustls::ClientConfig> {
    TLS_CONFIG
        .get_or_init(|| {
            let root_store =
                rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            let config = rustls::ClientConfig::builder()
                .with_root_certificates(root_store)
                .with_no_client_auth();
            Arc::new(config)
        })
        .clone()
}

use crate::stdlib_helpers::{arg_to_i64, arg_to_str, insert_func, to_result_tuple};

/// Parse a URL into (host, port, path, is_https).
fn parse_url(url: &str) -> Option<(String, u16, String, bool)> {
    let url = url.trim();
    let (without_scheme, is_https) = if let Some(rest) = url.strip_prefix("https://") {
        (rest, true)
    } else if let Some(rest) = url.strip_prefix("http://") {
        (rest, false)
    } else {
        (url, false)
    };

    let (host_port, path) = match without_scheme.find('/') {
        Some(idx) => (&without_scheme[..idx], &without_scheme[idx..]),
        None => (without_scheme, "/"),
    };

    let default_port: u16 = if is_https { 443 } else { 80 };
    let (host, port) = match host_port.find(':') {
        Some(idx) => {
            let port = host_port[idx + 1..].parse::<u16>().unwrap_or(default_port);
            (&host_port[..idx], port)
        }
        None => (host_port, default_port),
    };

    Some((host.to_string(), port, path.to_string(), is_https))
}

/// Make an HTTP/HTTPS request and return the response body.
/// Supports custom headers via `extra_headers` (e.g. `&[("Authorization", "Bearer ...")]`).
/// Timeout in seconds (default 30 for normal HTTP, callers can override).
pub(crate) fn http_request(method: &str, url: &str, body: Option<&str>) -> Result<String, String> {
    http_request_with_headers(method, url, body, &[], 30)
}

/// Full HTTP/HTTPS request with custom headers and configurable timeout.
pub(crate) fn http_request_with_headers(
    method: &str,
    url: &str,
    body: Option<&str>,
    extra_headers: &[(&str, &str)],
    timeout_secs: u64,
) -> Result<String, String> {
    let (host, port, path, is_https) = parse_url(url).ok_or_else(|| "Invalid URL".to_string())?;

    let tcp = TcpStream::connect(format!("{}:{}", host, port))
        .map_err(|e| format!("Connection failed: {}", e))?;

    tcp.set_read_timeout(Some(std::time::Duration::from_secs(timeout_secs)))
        .ok();

    // Build request
    let body_str = body.unwrap_or("");
    let mut request = if body_str.is_empty() {
        format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n",
            method, path, host
        )
    } else {
        format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
            method, path, host, body_str.len()
        )
    };

    // Append custom headers
    for (key, val) in extra_headers {
        request.push_str(&format!("{}: {}\r\n", key, val));
    }
    request.push_str("\r\n");
    if !body_str.is_empty() {
        request.push_str(body_str);
    }

    // Read response via plain TCP or TLS
    let response = if is_https {
        let config = get_tls_config();
        let server_name = rustls::pki_types::ServerName::try_from(host.clone())
            .map_err(|_| format!("Invalid hostname: {}", host))?;
        let conn = rustls::ClientConnection::new(config, server_name)
            .map_err(|e| format!("TLS handshake failed: {}", e))?;
        let mut tls = rustls::StreamOwned::new(conn, tcp);
        tls.write_all(request.as_bytes())
            .map_err(|e| format!("TLS write failed: {}", e))?;
        tls.flush()
            .map_err(|e| format!("TLS flush failed: {}", e))?;
        let mut buf = Vec::new();
        tls.read_to_end(&mut buf)
            .map_err(|e| format!("TLS read failed: {}", e))?;
        buf
    } else {
        let mut tcp = tcp;
        tcp.write_all(request.as_bytes())
            .map_err(|e| format!("Write failed: {}", e))?;
        tcp.flush().map_err(|e| format!("Flush failed: {}", e))?;
        let mut buf = Vec::new();
        tcp.read_to_end(&mut buf)
            .map_err(|e| format!("Read failed: {}", e))?;
        buf
    };

    let response_str = String::from_utf8_lossy(&response).to_string();
    parse_http_response(&response_str)
}

/// Parse an HTTP response: extract status, handle chunked encoding, return body.
fn parse_http_response(response_str: &str) -> Result<String, String> {
    if let Some(idx) = response_str.find("\r\n\r\n") {
        let headers = &response_str[..idx];
        let body = &response_str[idx + 4..];

        // Check status code
        let mut status_code: u16 = 200;
        if let Some(status_line) = headers.lines().next() {
            let parts: Vec<&str> = status_line.split_whitespace().collect();
            if parts.len() >= 2 {
                status_code = parts[1].parse().unwrap_or(0);
            }
        }

        // Decode body (handle chunked transfer-encoding)
        let headers_lower = headers.to_lowercase();
        let final_body = if headers_lower.contains("transfer-encoding: chunked") {
            decode_chunked(body)
        } else {
            body.to_string()
        };

        if status_code >= 400 {
            // Include body in error for API error messages
            if final_body.is_empty() {
                return Err(format!("HTTP {}", status_code));
            }
            return Err(format!("HTTP {}: {}", status_code, final_body));
        }

        Ok(final_body)
    } else {
        Ok(response_str.to_string())
    }
}

/// Decode HTTP chunked transfer-encoding.
fn decode_chunked(body: &str) -> String {
    let mut result = String::new();
    let mut remaining = body;
    while let Some(size_end) = remaining.find("\r\n") {
        let size_str = remaining[..size_end].trim();
        // Strip chunk extensions (after semicolon)
        let size_hex = size_str.split(';').next().unwrap_or("0").trim();
        let chunk_size = match usize::from_str_radix(size_hex, 16) {
            Ok(0) => break, // terminal chunk
            Ok(s) => s,
            Err(_) => break,
        };
        let data_start = size_end + 2;
        if data_start + chunk_size > remaining.len() {
            result.push_str(&remaining[data_start..]);
            break;
        }
        result.push_str(&remaining[data_start..data_start + chunk_size]);
        remaining = &remaining[data_start + chunk_size..];
        if remaining.starts_with("\r\n") {
            remaining = &remaining[2..];
        }
    }
    result
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
                    // Call handler with request map as TokValue arg.
                    // Cranelift closures return (tag, data) as two i64s, not a TokValue struct.
                    let req_val = TokValue::from_map(req_map);
                    let call: extern "C" fn(*mut u8, i64, i64) -> (i64, i64) =
                        unsafe { std::mem::transmute(fn_ptr) };
                    let (rtag, rdata) = call(env_ptr, req_val.tag as i64, unsafe {
                        req_val.data._raw as i64
                    });
                    let result = TokValue::from_tag_data(rtag, rdata);
                    // Parse result: string → body, map → {status, body}
                    let pair = match result.tag {
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
                    };
                    // Free handler result now that we've extracted the response data
                    result.rc_dec();
                    pair
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

        // Free request map and its contents (headers_map is nested inside)
        let req_val = TokValue::from_map(req_map);
        req_val.rc_dec();
    }

    TokValue::nil()
}

// ═══════════════════════════════════════════════════════════════
// Module constructor
// ═══════════════════════════════════════════════════════════════

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
