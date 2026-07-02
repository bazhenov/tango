use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::{
    hash::{DefaultHasher, Hasher},
    hint::black_box,
};
use tango_bench::{IntoBenchmarks, benchmark_fn, tango_benchmarks};

pub const INPUT_STRING: &str = include_str!("../data/input.txt");

fn benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("str", |b| {
            b.iter(|| black_box(INPUT_STRING).chars().count())
        }),
        benchmark_fn("binary_search", |b| {
            let mut vec = vec![0u64; 10 * 1024];
            let mut rand = SmallRng::seed_from_u64(42);
            rand.fill(&mut vec);
            vec.sort();

            b.iter(move || {
                let needle = rand.random::<u64>();
                vec.binary_search(&needle)
            })
        }),
        benchmark_fn("hash", |b| {
            b.iter(|| {
                let mut hasher = DefaultHasher::new();
                for i in black_box(0..100000) {
                    hasher.write_u64(i);
                }
                hasher.finish()
            })
        }),
    ]
}

tango_benchmarks!(benchmarks());
