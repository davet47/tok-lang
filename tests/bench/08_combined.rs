// Benchmark 08: Combined — realistic data processing pipeline
// Exercises: loops, arrays, maps, closures, iterators, filter, reduce, strings

use std::collections::HashMap;
use std::time::Instant;

fn lcg(seed: i64) -> i64 {
    (seed.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648
}

struct Record {
    name: String,
    score: i64,
    category: String,
}

struct Stats {
    count: usize,
    avg: i64,
    min: i64,
    max: i64,
}

fn main() {
    let n_records = 500_000;

    // Build dataset
    let build_start = Instant::now();
    let mut records: Vec<Record> = Vec::new();
    let mut seed: i64 = 42;
    for i in 0..n_records {
        seed = lcg(seed);
        let score = seed % 100;
        let cat = score % 4;
        let category = match cat {
            0 => "alpha",
            1 => "beta",
            2 => "gamma",
            _ => "delta",
        };
        records.push(Record {
            name: format!("item{}", i),
            score,
            category: category.to_string(),
        });
    }
    let build_ms = build_start.elapsed().as_millis();

    // Filter: keep records with score >= 50
    let filt_start = Instant::now();
    let high_scores: Vec<&Record> = records.iter().filter(|r| r.score >= 50).collect();
    let filt_ms = filt_start.elapsed().as_millis();

    // Transform: double scores
    let trans_start = Instant::now();
    let transformed: Vec<Record> = high_scores
        .iter()
        .map(|r| Record {
            name: r.name.clone(),
            score: r.score * 2,
            category: r.category.clone(),
        })
        .collect();
    let trans_ms = trans_start.elapsed().as_millis();

    // Group by category
    let group_start = Instant::now();
    let mut groups: HashMap<String, Vec<i64>> = HashMap::new();
    for r in &transformed {
        groups.entry(r.category.clone()).or_default().push(r.score);
    }
    let group_ms = group_start.elapsed().as_millis();

    // Sort each group and compute stats
    let sort_start = Instant::now();
    let mut stats: HashMap<String, Stats> = HashMap::new();
    for (k, scores) in &mut groups {
        scores.sort();
        let count = scores.len();
        let avg = scores.iter().sum::<i64>() / count as i64;
        let min = scores[0];
        let max = scores[count - 1];
        stats.insert(k.clone(), Stats { count, avg, min, max });
    }
    let sort_ms = sort_start.elapsed().as_millis();

    // Format output
    let fmt_start = Instant::now();
    let mut output = String::new();
    let mut stat_keys: Vec<&String> = stats.keys().collect();
    stat_keys.sort();
    for k in &stat_keys {
        let s = &stats[*k];
        output.push_str(&format!(
            "{}: count={} avg={} min={} max={}\n",
            k, s.count, s.avg, s.min, s.max
        ));
    }
    let fmt_ms = fmt_start.elapsed().as_millis();

    let total_ms = build_ms + filt_ms + trans_ms + group_ms + sort_ms + fmt_ms;
    println!("[bench] combined: {} records processed in {} ms", n_records, total_ms);
    println!(
        "[bench] combined_detail: build={}ms filter={}ms transform={}ms group={}ms sort={}ms format={}ms",
        build_ms, filt_ms, trans_ms, group_ms, sort_ms, fmt_ms
    );
    println!(
        "[bench] combined_counts: total={} filtered={} groups={}",
        records.len(),
        transformed.len(),
        stat_keys.len()
    );
    print!("{}", output);
}
