#![cfg_attr(feature = "align", feature(fn_align))]

use crate::test_funcs::{factorial, sum};
use std::rc::Rc;
use tango_bench::{
    async_benchmark_fn, asynchronous::tokio::TokioRuntime, benchmark_fn, tango_benchmarks,
    tango_main, IntoBenchmarks,
};
use test_funcs::{
    create_str_benchmark, sort_unstable, str_count, str_take, vec_benchmarks, IndexedString,
    INPUT_TEXT,
};

mod test_funcs;

fn num_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("sum", |b| b.iter(|| sum(4950))),
        benchmark_fn("factorial", |b| b.iter(|| factorial(495))),
        async_benchmark_fn("factorial_async", TokioRuntime, |b| {
            b.iter(|| async { factorial(495) })
        }),
    ]
}

fn str_benchmarks() -> impl IntoBenchmarks {
    let input = Rc::new(IndexedString::from(INPUT_TEXT));
    [
        create_str_benchmark("str_length/random", &input, str_count),
        create_str_benchmark("str_length/random_limited", &input, |s| str_take(4950, s)),
    ]
}

tango_benchmarks!(
    str_benchmarks(),
    num_benchmarks(),
    vec_benchmarks(sort_unstable)
);
tango_main!();
