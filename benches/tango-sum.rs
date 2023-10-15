use rust_pairwise_testing::{benchmark_fn, cli::run, Benchmark, StaticValue};
use test_funcs::sum;

mod test_funcs;

fn main() {
    let mut benchmark = Benchmark::new();

    benchmark.add_pair(
        benchmark_fn("sum_50000", |_| sum(50000)),
        benchmark_fn("sum_49500", |_| sum(49500)),
    );

    run(benchmark, &mut StaticValue(()));
}
