#![cfg_attr(feature = "align", feature(fn_align))]

use crate::test_funcs::factorial;
use std::time::Duration;
use tango_bench::{
    async_benchmark_fn, asynchronous::tokio::TokioRuntime, tango_benchmarks, tango_main,
    IntoBenchmarks,
};

mod test_funcs;

fn num_benchmarks() -> impl IntoBenchmarks {
    [
        async_benchmark_fn("factorial_async", TokioRuntime, |b| {
            b.iter(|| async { factorial(500) })
        }),
        async_benchmark_fn("sleep", TokioRuntime, |b| {
            b.iter(|| async { tokio::time::sleep(Duration::from_millis(1)).await })
        }),
    ]
}

tango_benchmarks!(num_benchmarks());
tango_main!();
