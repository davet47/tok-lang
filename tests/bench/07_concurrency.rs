// Benchmark 07: Concurrency — threads, channels, parallel map
// Measures thread spawn/join, channel throughput, parallel speedup

use std::sync::mpsc;
use std::thread;
use std::time::Instant;

fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() {
    // Part A: Parallel threads (8x fib(27))
    let start = Instant::now();
    let handles: Vec<_> = (0..8)
        .map(|_| thread::spawn(|| fib(27)))
        .collect();
    let r: i64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let par_ms = start.elapsed().as_millis();
    println!("[bench] concurrency_parallel: 8 threads fib(27) in {} ms (result={})", par_ms, r);

    // Part B: Channel throughput
    let n_msgs = 10_000_000;
    let (tx, rx) = mpsc::sync_channel::<i64>(100);
    let start = Instant::now();
    thread::spawn(move || {
        for i in 0..n_msgs as i64 {
            tx.send(i).unwrap();
        }
    });
    let mut total: i64 = 0;
    for _ in 0..n_msgs {
        total += rx.recv().unwrap();
    }
    let elapsed = start.elapsed().as_millis();
    println!("[bench] concurrency_channel: {} messages in {} ms (sum={})", n_msgs, elapsed, total);

    // Part C: Parallel map (8x fib(27))
    let start = Instant::now();
    let handles: Vec<_> = vec![27i64; 8]
        .into_iter()
        .map(|n| thread::spawn(move || fib(n)))
        .collect();
    let results: Vec<i64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let pmap_ms = start.elapsed().as_millis();
    let r2: i64 = results.iter().sum();
    println!("[bench] concurrency_pmap: 8 elements in {} ms (result={})", pmap_ms, r2);

    // Part D: Sequential comparison
    let start = Instant::now();
    let mut total2: i64 = 0;
    for _ in 0..8 {
        total2 += fib(27);
    }
    let seq_ms = start.elapsed().as_millis();
    println!("[bench] concurrency_sequential: 8 calls in {} ms (result={})", seq_ms, total2);

    // Speedup ratio
    let speedup = seq_ms as f64 / par_ms as f64;
    println!("[bench] concurrency_speedup: {:.1}x (parallel vs sequential)", speedup);
}
