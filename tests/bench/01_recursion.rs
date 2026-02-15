// Benchmark 01: Recursion — naive recursive Fibonacci
// Measures function call overhead, stack frame creation

use std::time::Instant;

fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() {
    let n = 45;
    let start = Instant::now();
    let r = fib(n);
    let elapsed = start.elapsed().as_millis();
    println!("[bench] recursion: fib({}) = {} in {} ms", n, r, elapsed);
}
