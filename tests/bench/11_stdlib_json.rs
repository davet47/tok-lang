// Benchmark 11: Stdlib JSON — parse + stringify
// Measures JSON serialization/deserialization performance

use std::time::Instant;

fn main() {
    // Part A: Parse simple array repeatedly
    let n_parse = 1_000_000;
    let start = Instant::now();
    let mut last: serde_json::Value = serde_json::Value::Null;
    for _ in 0..n_parse {
        last = serde_json::from_str("[1, 2, 3, 4, 5]").unwrap();
    }
    let len = last.as_array().map_or(0, |a| a.len());
    let elapsed = start.elapsed().as_millis();
    println!("[bench] json_parse_array: {} calls in {} ms (len={})", n_parse, elapsed, len);

    // Part B: Stringify map repeatedly
    let n_stringify = 1_000_000;
    let data = serde_json::json!({
        "name": "tok",
        "version": 1,
        "fast": true,
        "tags": ["lang", "compiled"]
    });
    let start = Instant::now();
    let mut last_s = String::new();
    for _ in 0..n_stringify {
        last_s = serde_json::to_string(&data).unwrap();
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] json_stringify: {} calls in {} ms (len={})", n_stringify, elapsed, last_s.len());

    // Part C: Roundtrip parse + stringify
    let n_roundtrip = 500_000;
    let json_str = r#"{"x":1,"y":2,"z":3}"#;
    let start = Instant::now();
    let mut last_rt = String::new();
    for _ in 0..n_roundtrip {
        let obj: serde_json::Value = serde_json::from_str(json_str).unwrap();
        last_rt = serde_json::to_string(&obj).unwrap();
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] json_roundtrip: {} parse+stringify in {} ms (last={})", n_roundtrip, elapsed, last_rt);
}
