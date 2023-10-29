#![cfg_attr(feature = "align", feature(fn_align))]

mod test_funcs;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use tango_bench::Generator;
use test_funcs::{factorial, str_count, str_count_rev, str_take, sum, RandomString};

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

    group.bench_function("str_length_4950", |b| {
        let mut generator = RandomString::new().unwrap();
        b.iter_batched(
            || generator.next_haystack(),
            |s| str_take(4950, &s, &()),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("str_length_5000", |b| {
        let mut generator = RandomString::new().unwrap();
        b.iter_batched(
            || generator.next_haystack(),
            |s| str_take(5000, &s, &()),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("std_count", |b| {
        let mut generator = RandomString::new().unwrap();
        b.iter_batched(
            || generator.next_haystack(),
            |s| str_count(&s, &()),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("std_count_rev", |b| {
        let mut generator = RandomString::new().unwrap();
        b.iter_batched(
            || generator.next_haystack(),
            |s| str_count_rev(&s, &()),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, utf8_benchmark, sum_benchmark);
criterion_main!(benches);
