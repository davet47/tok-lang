// Benchmark 03: Arrays — build, sort, filter, reduce
// Measures array allocation, sorting, functional operations

use std::time::Instant;

fn main() {
    // Part A: Build array via push
    let n_build = 2_000_000;
    let start = Instant::now();
    let mut a: Vec<i64> = Vec::new();
    for i in 0..n_build {
        a.push(i);
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] array_build: {} elements in {} ms", n_build, elapsed);

    // Part B: Sort (reversed array)
    let mut b: Vec<i64> = a.iter().rev().copied().collect();
    let n = b.len();
    let start = Instant::now();
    b.sort();
    let elapsed = start.elapsed().as_millis();
    println!("[bench] array_sort: {} elements in {} ms (first={} last={})", n, elapsed, b[0], b[n - 1]);

    // Part C: Filter + Reduce
    let start = Instant::now();
    let total: i64 = a.iter().filter(|x| *x % 2 == 0).sum();
    let elapsed = start.elapsed().as_millis();
    println!("[bench] array_filter_reduce: {} elements in {} ms (result={})", a.len(), elapsed, total);
}
