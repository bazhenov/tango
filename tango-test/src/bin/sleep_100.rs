use std::{thread, time::Duration};
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main};

tango_benchmarks!([benchmark_fn("sleep", |b| b.iter(|| {
    thread::sleep(Duration::from_millis(100));
}))]);
tango_main!();
