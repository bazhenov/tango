#![cfg_attr(feature = "align", feature(fn_align))]
use rust_pairwise_testing::{benchmark_fn, cli::run, Benchmark};
use test_funcs::{std, std_count, std_count_rev, std_take, RandomStringGenerator};

mod test_funcs;

fn main() {
    let mut payloads = RandomStringGenerator::new().unwrap();
    let mut benchmark = Benchmark::new();

    benchmark.add_pair(benchmark_fn("std", std), benchmark_fn("std", std));
    benchmark.add_pair(
        benchmark_fn("std", std),
        benchmark_fn("std_count", std_count),
    );
    benchmark.add_pair(
        benchmark_fn("std_count", std_count),
        benchmark_fn("std_count_rev", std_count_rev),
    );
    benchmark.add_pair(
        benchmark_fn("std_5000", std_take::<5000>),
        benchmark_fn("std_4950", std_take::<4950>),
    );

    run(benchmark, &mut [&mut payloads]);
}
