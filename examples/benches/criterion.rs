#![cfg_attr(feature = "align", feature(fn_align))]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use tango_bench::Generator;
use test_funcs::{factorial, str_take, sum, RandomSubstring};

mod test_funcs;

/// Because benchmarks are builded with linker flag -rdynamic there should be dummy library entrypoint defined
/// in all benchmarks. This is only needed when two benchmarks harnesses are used in a single crate.
mod dummy_entrypoint {
    tango_bench::tango_benchmarks!([]);
}

fn sum_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic");

    group.bench_function("sum_5000", |b| {
        b.iter(|| sum(5000));
    });

    group.bench_function("sum_4950", |b| {
        b.iter(|| sum(4950));
    });

    group.bench_function("factorial_500", |b| {
        b.iter(|| factorial(500));
    });

    group.bench_function("factorial_495", |b| {
        b.iter(|| factorial(495));
    });

    group.finish()
}

fn utf8_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("utf8");

    group.bench_function("str_length_4950", str_length::<4950>);
    group.bench_function("str_length_5000", str_length::<5000>);
}

fn str_length<const N: usize>(b: &mut criterion::Bencher<'_>) {
    let mut generator = RandomSubstring::new();
    let string = generator.next_haystack();
    b.iter_batched(
        || generator.next_needle(&string),
        |offset| str_take(N, &string, &offset),
        BatchSize::SmallInput,
    );
}

criterion_group!(benches, utf8_benchmark, sum_benchmark);
criterion_main!(benches);
