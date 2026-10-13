use tango_bench::{benchmark_fn, tango_benchmarks};

tango_benchmarks!([
    benchmark_fn("baseline/compression", |b| b.iter(|| { 1 + 1 })),
    benchmark_fn("candidate/compression", |b| b.iter(|| { 1 + 1 })),
    benchmark_fn("candidate/candidate-only", |b| b.iter(|| { 1 + 1 }))
]);
