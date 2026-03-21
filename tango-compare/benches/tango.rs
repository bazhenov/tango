use std::hint::black_box;
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main};

pub const INPUT_STRING: &str = include_str!("../data/input.txt");

tango_benchmarks!([benchmark_fn("str", |b| {
    b.iter(|| black_box(INPUT_STRING).chars().count())
})]);
tango_main!();
