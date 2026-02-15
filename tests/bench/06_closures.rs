// Benchmark 06: Closures — lambda invoke, closure capture, higher-order functions
// Measures closure call overhead, captured variable access, function-as-argument

use std::time::Instant;
use std::hint::black_box;

fn apply(f: &dyn Fn(i64) -> i64, x: i64) -> i64 {
    f(x)
}

fn main() {
    // Part A: Lambda invocation
    let double = |x: i64| x * 2;
    let n_invoke: i64 = 200_000_000;
    let start = Instant::now();
    let mut x: i64 = 0;
    for i in 0..n_invoke {
        x = double(i);
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] closure_invoke: {} calls in {} ms (result={})", n_invoke, elapsed, black_box(x));

    // Part B: Closure capture
    let offset: i64 = 42;
    let adder = |x: i64| x + offset;
    let n_capture: i64 = 200_000_000;
    let start = Instant::now();
    let mut x: i64 = 0;
    for i in 0..n_capture {
        x = adder(i);
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] closure_capture: {} calls in {} ms (result={})", n_capture, elapsed, black_box(x));

    // Part C: Higher-order function
    let inc = |x: i64| x + 1;
    let n_hof: i64 = 100_000_000;
    let start = Instant::now();
    let mut x: i64 = 0;
    for i in 0..n_hof {
        x = apply(&inc, i);
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] closure_hof: {} calls in {} ms (result={})", n_hof, elapsed, black_box(x));
}
