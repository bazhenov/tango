use tango_bench::{benchmark_fn, tango_benchmarks};

tango_benchmarks!([
    benchmark_fn("bench1", |b| b.iter(|| { 1 + 1 })),
    benchmark_fn("bench2", |b| b.iter(|| { 1 + 1 }))
]);
