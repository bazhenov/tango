use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

pub const INPUT_STRING: &str = include_str!("../data/input.txt");

pub fn str_length(s: &str) -> usize {
    s.chars().count()
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("str", |b| {
        b.iter(|| black_box(INPUT_STRING).chars().count())
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
