// Benchmark 02: Loops — iteration overhead
// Measures counting loop, arithmetic body, while loop

use std::time::Instant;
use std::hint::black_box;

fn main() {
    // Part A: Counting loop
    let n_count: i64 = 5_000_000_000;
    let start = Instant::now();
    let mut x: i64 = 0;
    for _ in 0..n_count {
        x += 1;
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] loops_count: {} iterations in {} ms (result={})", n_count, elapsed, black_box(x));

    // Part B: Arithmetic body
    let n_arith: i64 = 1_000_000_000;
    let start = Instant::now();
    let mut s: i64 = 0;
    for i in 0..n_arith {
        s += i * i % 7;
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] loops_arith: {} iterations in {} ms (result={})", n_arith, elapsed, black_box(s));

    // Part C: While loop
    let n_while: i64 = 500_000_000;
    let start = Instant::now();
    let mut n = n_while;
    while n > 0 {
        n -= 1;
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] loops_while: {} iterations in {} ms (result={})", n_while, elapsed, black_box(n));
}
