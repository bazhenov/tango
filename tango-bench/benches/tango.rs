#![cfg_attr(feature = "align", feature(fn_align))]

use rand::{distributions::Standard, rngs::SmallRng, Rng, SeedableRng};
use tango_bench::{
    benchmark_fn, iqr_variance_thresholds, tango_benchmarks, tango_main, IntoBenchmarks, Summary,
};

fn summary_benchmarks() -> impl IntoBenchmarks {
    [benchmark_fn("summary", move |b| {
        let rnd = SmallRng::seed_from_u64(b.seed);
        let input: Vec<i64> = rnd.sample_iter(Standard).take(1000).collect();
        b.iter(move || Summary::from(&input))
    })]
}

fn iqr_interquartile_range_benchmarks() -> impl IntoBenchmarks {
    [benchmark_fn("iqr", move |b| {
        let rnd = SmallRng::seed_from_u64(b.seed);
        let input: Vec<f64> = rnd.sample_iter(Standard).take(1000).collect();
        b.iter(move || iqr_variance_thresholds(input.clone()))
    })]
}

fn empty_benchmarks() -> impl IntoBenchmarks {
    [benchmark_fn("measure_empty_function", move |p| {
        let mut bench = benchmark_fn("_", |b| b.iter(|| 42));
        let mut state = bench.prepare_state(p.seed);
        p.iter(move || state.measure(1))
    })]
}

tango_benchmarks!(
    empty_benchmarks(),
    summary_benchmarks(),
    iqr_interquartile_range_benchmarks()
);
tango_main!();
