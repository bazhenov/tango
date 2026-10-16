use std::hint::black_box;

use tango_bench::{aux, AuxMetricEntry};

fn main() {
    let metric = aux::user_time();
    let mut counts = [0; 13];
    let base = 10;

    for _ in 0..100 {
        let elapsed = measure(&metric);
        let magnitude: usize = if elapsed == 0 {
            0
        } else {
            elapsed.ilog(base) as usize + 1
        };
        if magnitude <= counts.len() {
            counts[magnitude] += 1;
        }
    }
    for (order, cnt) in counts.iter().enumerate() {
        let order = if order == 0 {
            "0".into()
        } else {
            format!("{base}^{}", order - 1)
        };
        println!("{order:5} : {cnt}");
    }
}

fn measure(metric: &AuxMetricEntry) -> u64 {
    let start = (metric.start)();
    // Do some CPU work
    let mut sum = 0u64;
    // black_box prevents constant folding optimization
    for i in 0..black_box(10_000_000) {
        sum = sum.wrapping_add(i);
    }
    black_box(sum);
    (metric.finish)(start)
}
