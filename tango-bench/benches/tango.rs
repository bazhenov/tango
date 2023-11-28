#![cfg_attr(feature = "align", feature(fn_align))]

use num_traits::ToPrimitive;
use tango_bench::{
    benchmark_fn, benchmarks, cli, BenchmarkMatrix, IntoBenchmarks, StaticValue, Summary,
};
use test_funcs::RandomVec;

mod test_funcs;

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn create_summary<T: Copy + Ord + Default, N>(input: &Vec<T>, _: &N) -> Option<Summary<T>>
where
    T: ToPrimitive,
{
    Summary::from(input)
}

fn summary_benchmarks() -> impl IntoBenchmarks {
    let generator = RandomVec::<i64>::new(1_000);
    BenchmarkMatrix::new(generator).add_function("summary", create_summary)
}

fn empty_benchmarks() -> impl IntoBenchmarks {
    vec![benchmark_fn("empty_bench", || ())]
}

fn generator_empty_benchmarks() -> impl IntoBenchmarks {
    let generator = StaticValue((), ());
    BenchmarkMatrix::new(generator).add_function("generator_empty_bench", |_, _| ())
}

benchmarks!(
    empty_benchmarks(),
    generator_empty_benchmarks(),
    summary_benchmarks()
);

fn main() {
    cli::run(Default::default())
}
