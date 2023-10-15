#![cfg_attr(feature = "align", feature(fn_align))]
use rand::{rngs::SmallRng, Rng};
use rust_pairwise_testing::{benchmark_fn, cli::run, Benchmark, Generator};
use std::num::NonZeroUsize;
use test_funcs::{std, std_4925, std_5000, std_count, std_count_rev, RandomStringGenerator};

mod test_funcs;

struct RandomVec(SmallRng, NonZeroUsize);

impl Generator for RandomVec {
    type Output = Vec<u32>;

    fn next_payload(&mut self) -> Self::Output {
        let RandomVec(rng, size) = self;
        let mut v = vec![0; (*size).into()];
        rng.fill(&mut v[..]);
        v
    }
}

fn main() {
    let mut payloads = RandomStringGenerator::new().unwrap();
    let mut benchmark = Benchmark::new();

    benchmark.add_pair(
        benchmark_fn("std", std),
        benchmark_fn("std_count", std_count),
    );
    benchmark.add_pair(
        benchmark_fn("std_count", std_count),
        benchmark_fn("std_count_rev", std_count_rev),
    );
    benchmark.add_pair(
        benchmark_fn("std_5000", std_5000),
        benchmark_fn("std_4925", std_4925),
    );

    run(benchmark, &mut payloads);
}
