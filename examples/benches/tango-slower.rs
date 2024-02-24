#![cfg_attr(feature = "align", feature(fn_align))]

use crate::test_funcs::{factorial, sum};
use tango_bench::{
    generators::RandomVec, new_api::new_bench, tango_benchmarks, tango_main, BenchmarkMatrix,
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
        new_bench("sum", |_| || sum(4950)),
        new_bench("factorial", |_| || factorial(495)),
    ]
}

fn vec_benchmarks() -> impl IntoBenchmarks {
    BenchmarkMatrix::with_params([100, 1_000, 10_000, 100_000], RandomVec::<u64>::new)
        .add_function("sort", sort_stable)
}

tango_benchmarks!(str_benchmarks(), num_benchmarks(), vec_benchmarks());
tango_main!();
