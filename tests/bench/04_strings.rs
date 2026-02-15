// Benchmark 04: Strings — concatenation, format, split/join
// Measures string allocation, formatting overhead, string ops

use std::time::Instant;

fn main() {
    // Part A: Concatenation
    let n_concat = 20_000_000;
    let start = Instant::now();
    let mut s = String::new();
    for _ in 0..n_concat {
        s.push('a');
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] string_concat: {} iterations in {} ms (len={})", n_concat, elapsed, s.len());

    // Part B: Interpolation (format!)
    let n_interp = 20_000_000;
    let start = Instant::now();
    let mut last = String::new();
    for i in 0..n_interp {
        last = format!("{}", i);
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] string_interp: {} iterations in {} ms (last={})", n_interp, elapsed, last);

    // Part C: Split + Join
    let base = "the quick brown fox jumps over the lazy dog";
    let n_split = 1_000_000;
    let start = Instant::now();
    let mut result = String::new();
    for _ in 0..n_split {
        let parts: Vec<&str> = base.split(' ').collect();
        result = parts.join("-");
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] string_split_join: {} iterations in {} ms (result={})", n_split, elapsed, result);
}
