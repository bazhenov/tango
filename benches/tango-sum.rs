use rust_pairwise_testing::{benchmark_fn, cli::run, Benchmark, StaticValue};
use test_funcs::sum;

mod test_funcs;

fn main() {
    let mut benchmark = Benchmark::new(StaticValue(()));

    benchmark.add_function("sum_50000", benchmark_fn(|_| sum(50000)));
    benchmark.add_function("sum_49500", benchmark_fn(|_| sum(49500)));

    run(benchmark)
}
