#![cfg_attr(feature = "align", feature(fn_align))]

use tango_bench::{
    benchmark_fn, cli, GeneratorBenchmarks, IntoBenchmarks, MeasureTarget, MeasurementSettings,
};
use test_funcs::{sort_unstable, str_count, str_take, RandomString, RandomVec};

use crate::test_funcs::{factorial, sum};

mod test_funcs;

pub fn str_benchmarks() -> impl IntoBenchmarks {
    let generator = RandomString::new().unwrap();
    let mut benchmarks = GeneratorBenchmarks::with_generator(generator);

    benchmarks
        .add("str_length", str_count)
        .add("str_length_limit", |h, n| str_take(4950, h, n));

    benchmarks
}

pub fn num_benchmarks() -> impl IntoBenchmarks {
    vec![
        benchmark_fn("sum", || sum(4950)),
        benchmark_fn("factorial", || factorial(495)),
    ]
}

pub fn vec_benchmarks() -> impl IntoBenchmarks {
    let mut benchmarks =
        GeneratorBenchmarks::with_generators([100, 1_000, 10_000, 100_000], |size| {
            RandomVec::<u64>::new(size)
        });

    benchmarks.add("sort", sort_unstable);

    benchmarks
}

#[no_mangle]
pub fn create_benchmarks() -> Vec<Box<dyn MeasureTarget>> {
    let mut benchmarks = vec![];
    benchmarks.extend(str_benchmarks().into_benchmarks());
    benchmarks.extend(num_benchmarks().into_benchmarks());
    benchmarks.extend(vec_benchmarks().into_benchmarks());
    benchmarks
}

fn main() {
    cli::run(MeasurementSettings::default())
}
