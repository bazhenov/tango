#![cfg_attr(feature = "align", feature(fn_align))]

use rust_pairwise_testing::{benchmark_fn, cli, Benchmark, Summary};
use test_funcs::RandomVec;

mod test_funcs;

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn old_summary<T: Copy + Ord>(input: &Vec<T>) -> Option<Summary<T>>
where
    i64: From<T>,
{
    Summary::from(input)
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn new_summary<T: Copy + Ord>(input: &Vec<T>) -> Option<Summary<T>>
where
    i64: From<T>,
{
    Summary::from(input)
}

fn main() {
    let mut payloads = RandomVec::<i64>::new(1_000);

    let mut b = Benchmark::new();
    b.add_pair(
        benchmark_fn("old", old_summary),
        benchmark_fn("new", new_summary),
    );

    cli::run(b, &mut payloads)
}
