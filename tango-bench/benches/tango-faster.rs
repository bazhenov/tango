#![cfg_attr(feature = "align", feature(fn_align))]

use crate::test_funcs::{factorial, sum};
use std::process::ExitCode;
use tango_bench::{
    benchmark_fn, benchmarks, cli, BenchmarkMatrix, IntoBenchmarks, MeasurementSettings,
};
use test_funcs::{sort_unstable, str_count, str_take, RandomString, RandomVec};

mod test_funcs;

fn str_benchmarks() -> impl IntoBenchmarks {
    BenchmarkMatrix::new(RandomString::new())
        .add_function("str_length", str_count)
        .add_function("str_length_limit", |h, n| str_take(4950, h, n))
}

fn num_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("sum", || sum(4950)),
        benchmark_fn("factorial", || factorial(495)),
    ]
}

fn vec_benchmarks() -> impl IntoBenchmarks {
    BenchmarkMatrix::with_params([100, 1_000, 10_000, 100_000], RandomVec::<u64>::new)
        .add_function("sort", sort_unstable)
}

benchmarks!(str_benchmarks(), num_benchmarks(), vec_benchmarks());

fn main() -> tango_bench::cli::Result<ExitCode> {
    cli::run(MeasurementSettings::default())
}
