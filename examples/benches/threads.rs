use std::thread;
use tango_bench::{benchmark_fn, tango_benchmarks};

tango_benchmarks!([benchmark_fn("spawn_join", |b| {
    b.iter(|| thread::spawn(|| {}).join())
})]);
