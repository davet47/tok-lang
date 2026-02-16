// Benchmark 09: Stdlib Math — trigonometry, sqrt, pow, log
// Measures math function call overhead

use std::time::Instant;

fn main() {
    // Part A: sqrt loop
    let n_sqrt: i64 = 10_000_000;
    let start = Instant::now();
    let mut total: f64 = 0.0;
    for i in 1..=n_sqrt {
        total += (i as f64).sqrt();
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] math_sqrt: {} calls in {} ms (result={})", n_sqrt, elapsed, total.floor() as i64);

    // Part B: sin + cos
    let n_trig: i64 = 5_000_000;
    let start = Instant::now();
    let mut sum: f64 = 0.0;
    for i in 0..n_trig {
        let x = i as f64 * 0.001;
        sum += x.sin() + x.cos();
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] math_trig: {} sin+cos pairs in {} ms (result={})", n_trig, elapsed, sum.floor() as i64);

    // Part C: pow + log roundtrip
    let n_powlog: i64 = 5_000_000;
    let start = Instant::now();
    let mut acc: f64 = 0.0;
    for i in 1..=n_powlog {
        acc += ((2.0_f64).powf((i % 20) as f64) + 1.0).ln();
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] math_powlog: {} pow+log calls in {} ms (result={})", n_powlog, elapsed, acc.floor() as i64);
}
