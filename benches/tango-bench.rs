#![cfg_attr(feature = "align", feature(fn_align))]

use rust_pairwise_testing::{benchmark_fn, cli, Benchmark, Summary};
use test_funcs::RandomVec;

mod test_funcs;

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn old_summary(input: &Vec<u32>) -> Summary<u32> {
    Summary::from(input).unwrap()
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn new_summary(input: &Vec<u32>) -> Summary<u32> {
    Summary::from(input).unwrap()
}

fn main() {
    let mut payloads = RandomVec::new(1_000);

    let mut b = Benchmark::new();
    b.add_pair(
        benchmark_fn("old", old_summary),
        benchmark_fn("new", new_summary),
    );

    cli::run(b, &mut payloads)
}
