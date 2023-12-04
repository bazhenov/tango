#![cfg_attr(feature = "align", feature(fn_align))]

use crate::test_funcs::{factorial, sum};
use tango_bench::{
    benchmark_fn, generators::RandomVec, tango_benchmarks, tango_main, BenchmarkMatrix,
    IntoBenchmarks,
};
use test_funcs::{sort_stable, str_count_rev, str_take, RandomSubstring};

mod test_funcs;

fn str_benchmarks() -> impl IntoBenchmarks {
    BenchmarkMatrix::new(RandomSubstring::new())
        .add_function("str_length", str_count_rev)
        .add_function("str_length_limit", |h, n| str_take(5000, h, n))
}

fn num_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("sum", || sum(5000)),
        benchmark_fn("factorial", || factorial(500)),
    ]
}

fn vec_benchmarks() -> impl IntoBenchmarks {
    BenchmarkMatrix::with_params([100, 1_000, 10_000, 100_000], RandomVec::<u64>::new)
        .add_function("sort", sort_stable)
}

tango_benchmarks!(str_benchmarks(), num_benchmarks(), vec_benchmarks());
tango_main!();
