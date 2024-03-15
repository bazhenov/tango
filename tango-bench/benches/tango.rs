#![cfg_attr(feature = "align", feature(fn_align))]

use tango_bench::{
    benchmark_fn, generators::RandomVec, iqr_variance_thresholds, tango_benchmarks, tango_main,
    Generator, IntoBenchmarks, Summary,
};

#[derive(Clone)]
struct StaticValue<H, N>(
    /// Haystack value
    pub H,
    /// Needle value
    pub N,
);

impl<H: Clone, N: Copy> Generator for StaticValue<H, N> {
    type Haystack = H;
    type Needle = N;

    fn next_haystack(&mut self) -> Self::Haystack {
        self.0.clone()
    }

    fn next_needle(&mut self, _: &Self::Haystack) -> Self::Needle {
        self.1
    }

    fn name(&self) -> &str {
        "StaticValue"
    }

    fn sync(&mut self, _: u64) {}
}

fn summary_benchmarks() -> impl IntoBenchmarks {
    let mut generator = RandomVec::<i64>::new(1_000);
    [benchmark_fn("summary", move |b| {
        generator.sync(b.seed);
        let input = generator.next_haystack();
        b.iter(move || Summary::from(&input))
    })]
}

fn iqr_interquartile_range_benchmarks() -> impl IntoBenchmarks {
    let mut generator = RandomVec::<f64>::new(1_000);
    [benchmark_fn("iqr", move |b| {
        generator.sync(b.seed);
        let input = generator.next_haystack();
        b.iter(move || iqr_variance_thresholds(input.clone()))
    })]
}

fn empty_benchmarks() -> impl IntoBenchmarks {
    [benchmark_fn("measure_empty_function", move |p| {
        let mut bench = benchmark_fn("_", |b| b.iter(|| 42));
        bench.prepare_state(p.seed);
        p.iter(move || bench.measure(1))
    })]
}

tango_benchmarks!(
    empty_benchmarks(),
    summary_benchmarks(),
    iqr_interquartile_range_benchmarks()
);
tango_main!();
