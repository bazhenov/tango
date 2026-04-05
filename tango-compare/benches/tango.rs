use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::{hint::black_box, usize};
use tango_bench::{
    IntoBenchmarks, MeasurementSettings, SampleLengthKind, benchmark_fn, tango_benchmarks,
    tango_main,
};

pub const INPUT_STRING: &str = include_str!("../data/input.txt");

fn benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("str", |b| {
            b.iter(|| black_box(INPUT_STRING).chars().count())
        }),
        benchmark_fn("binary_search", |b| {
            let mut vec = vec![0u64; 1024 * 1024];
            let mut rand = SmallRng::seed_from_u64(42);
            rand.fill(&mut vec);
            vec.sort();

            b.iter(move || {
                let needle = rand.random::<u64>();
                vec.binary_search(&needle)
            })
        }),
    ]
}

tango_benchmarks!(benchmarks());
tango_main!(MeasurementSettings {
    samples_per_haystack: usize::MAX,
    sampler_type: SampleLengthKind::Flat,
    ..Default::default()
});
