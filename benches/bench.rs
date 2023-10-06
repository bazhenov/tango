use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rust_pairwise_testing::Generator;
use test_funcs::{std_4925, std_5000, std_count, std_count_rev, RandomStringGenerator};

mod test_funcs;

fn sum(n: usize) -> usize {
    let mut sum = 0;
    for i in 0..black_box(n) {
        sum += black_box(i);
    }
    sum
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("utf8");

    group.bench_function("sum_50000", |b| {
        b.iter(|| sum(50000));
    });

    group.bench_function("sum_49250", |b| {
        b.iter(|| sum(49250));
    });

    group.bench_function("std_length_4925", |b| {
        let mut generator = RandomStringGenerator::new().unwrap();
        b.iter_batched(
            || generator.next_payload(),
            |s| std_4925(&s),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("std_length_5000", |b| {
        let mut generator = RandomStringGenerator::new().unwrap();
        b.iter_batched(
            || generator.next_payload(),
            |s| std_5000(&s),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("std_count", |b| {
        let mut generator = RandomStringGenerator::new().unwrap();
        b.iter_batched(
            || generator.next_payload(),
            |s| std_count(&s),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("std_count_rev", |b| {
        let mut generator = RandomStringGenerator::new().unwrap();
        b.iter_batched(
            || generator.next_payload(),
            |s| std_count_rev(&s),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
