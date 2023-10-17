#![cfg_attr(feature = "align", feature(fn_align))]
use rust_pairwise_testing::{benchmark_fn, cli::run, Benchmark, StaticValue};
use test_funcs::{factorial, sum};

mod test_funcs;

fn main() {
    let mut benchmark = Benchmark::new();

    benchmark.add_pair(
        benchmark_fn("sum_50000", |_| sum(5000)),
        benchmark_fn("sum_49500", |_| sum(4950)),
    );

    benchmark.add_pair(
        benchmark_fn("factorial_500", |_| factorial(500)),
        benchmark_fn("factorial_495", |_| factorial(495)),
    );

    run(benchmark, &mut StaticValue(()));
}
