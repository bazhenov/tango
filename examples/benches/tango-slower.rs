#![cfg_attr(feature = "align", feature(fn_align))]

use crate::test_funcs::{factorial, sum};
use rand::{distributions::Standard, rngs::SmallRng, Rng, SeedableRng};
use std::rc::Rc;
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main, IntoBenchmarks};
use test_funcs::{
    create_str_benchmark, sort_stable, str_count_rev, str_take, IndexedString, INPUT_TEXT,
};

mod test_funcs;

fn num_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("sum", |b| b.iter(|| sum(5000))),
        benchmark_fn("factorial", |b| b.iter(|| factorial(500))),
    ]
}

fn vec_benchmarks() -> impl IntoBenchmarks {
    let mut benches = vec![];
    for size in [100, 1_000, 10_000, 100_000] {
        benches.push(benchmark_fn(format!("sort/{}", size), move |b| {
            let rnd = SmallRng::seed_from_u64(b.seed);
            let input: Vec<u64> = rnd.sample_iter(Standard).take(1000).collect();
            b.iter(move || sort_stable(&input, &()))
        }))
    }
    benches
}

fn str_benchmarks() -> impl IntoBenchmarks {
    let input = Rc::new(IndexedString::from(INPUT_TEXT));
    [
        create_str_benchmark("str_length/random", &input, str_count_rev),
        create_str_benchmark("str_length/random_limited", &input, |s| str_take(5000, s)),
    ]
}

tango_benchmarks!(str_benchmarks(), num_benchmarks(), vec_benchmarks());
tango_main!();
