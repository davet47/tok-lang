// Benchmark 10: Stdlib Str — upper/lower, contains, replace, split
// Measures string manipulation overhead

use std::time::Instant;

fn main() {
    // Part A: upper + lower case conversion loop
    let n_case = 2_000_000;
    let start = Instant::now();
    let mut last = String::new();
    for _ in 0..n_case {
        last = "hello world".to_uppercase();
        last = last.to_lowercase();
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] str_case: {} upper+lower pairs in {} ms (last={})", n_case, elapsed, last);

    // Part B: contains + starts_with + ends_with checks
    let base = "the quick brown fox jumps over the lazy dog";
    let n_search = 2_000_000;
    let start = Instant::now();
    let mut count: i64 = 0;
    for _ in 0..n_search {
        if base.contains("fox") { count += 1; }
        if base.starts_with("the") { count += 1; }
        if base.ends_with("dog") { count += 1; }
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] str_search: {} x3 checks in {} ms (count={})", n_search, elapsed, count);

    // Part C: replace
    let n_replace = 1_000_000;
    let start = Instant::now();
    let mut last2 = String::new();
    for _ in 0..n_replace {
        last2 = "hello world hello".replace("hello", "hi");
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] str_replace: {} calls in {} ms (last={})", n_replace, elapsed, last2);

    // Part D: split + rejoin
    let n_split = 1_000_000;
    let start = Instant::now();
    let mut last3 = String::new();
    for _ in 0..n_split {
        let parts: Vec<&str> = "a,b,c,d,e".split(',').collect();
        last3 = parts.join("-");
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] str_split: {} split+join in {} ms (last={})", n_split, elapsed, last3);
}
