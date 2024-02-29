#![cfg_attr(feature = "align", feature(fn_align))]

use crate::test_funcs::{factorial, sum};
use std::rc::Rc;
use tango_bench::{
    benchmark_fn, generators::RandomVec, tango_benchmarks, tango_main, BenchmarkMatrix,
    IntoBenchmarks,
};
use test_funcs::{
    create_str_benchmark, sort_unstable, str_count, str_take, IndexedString, INPUT_TEXT,
};

mod test_funcs;

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

fn str_benchmarks() -> impl IntoBenchmarks {
    let input = Rc::new(IndexedString::from(INPUT_TEXT));
    [
        create_str_benchmark("str_length/random", &input, str_count),
        create_str_benchmark("str_length/random_limited", &input, |s| str_take(4950, s)),
    ]
}

tango_benchmarks!(str_benchmarks(), num_benchmarks(), vec_benchmarks());
tango_main!();
