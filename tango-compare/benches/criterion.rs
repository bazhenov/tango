use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::{
    hash::{DefaultHasher, Hasher},
    hint::black_box,
};

pub const INPUT_STRING: &str = include_str!("../data/input.txt");

pub fn str_length(s: &str) -> usize {
    s.chars().count()
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("str", |b| {
        b.iter(|| black_box(INPUT_STRING).chars().count())
    });

    let mut vec = vec![0u64; 10 * 1024];
    let mut rand = SmallRng::seed_from_u64(42);
    rand.fill(&mut vec);
    vec.sort();

    c.bench_function("binary_search", |b| {
        let needle = rand.random::<u64>();
        b.iter(|| vec.binary_search(&needle))
    });

    c.bench_function("hash", |b| {
        b.iter(|| {
            let mut hasher = DefaultHasher::new();
            for i in black_box(0..100000) {
                hasher.write_u64(i);
            }
            hasher.finish()
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
