use std::hint::black_box;
use std::thread;
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main};

tango_benchmarks!([benchmark_fn("spawn_join", |b| {
    b.iter(|| thread::spawn(|| black_box(1)).join())
})]);
tango_main!();
