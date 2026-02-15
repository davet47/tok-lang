// Benchmark 05: Maps — build, lookup, membership
// Measures HashMap insertion, O(1) lookup, contains_key

use std::collections::HashMap;
use std::time::Instant;

fn main() {
    // Part A: Build map
    let n_build = 500_000;
    let start = Instant::now();
    let mut m: HashMap<String, i64> = HashMap::new();
    for i in 0..n_build {
        m.insert(format!("k{}", i), i);
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] map_build: {} entries in {} ms", n_build, elapsed);

    // Part B: Lookup every key
    let start = Instant::now();
    let mut total: i64 = 0;
    for i in 0..n_build {
        total += m[&format!("k{}", i)];
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] map_lookup: {} lookups in {} ms (sum={})", n_build, elapsed, total);

    // Part C: contains_key membership
    let n_has = 1_000_000;
    let start = Instant::now();
    let mut found: i64 = 0;
    for i in 0..n_has {
        let k = format!("k{}", i);
        if m.contains_key(&k) {
            found += 1;
        }
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] map_has: {} checks in {} ms (found={})", n_has, elapsed, found);
}
