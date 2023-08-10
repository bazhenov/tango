use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rust_pairwise_testing::{
    test_funcs::{std_4925, std_5000, std_count, std_count_rev},
    Generator, RandomStringGenerator,
};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("utf8");

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
