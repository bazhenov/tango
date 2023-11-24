#![cfg_attr(feature = "align", feature(fn_align))]

use num_traits::ToPrimitive;
use tango_bench::{_benchmark_fn, cli, Benchmark, Summary};
use test_funcs::RandomVec;

mod test_funcs;

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn old_summary<T: Copy + Ord + Default, N>(input: &Vec<T>, _: &N) -> Option<Summary<T>>
where
    T: ToPrimitive,
{
    Summary::from(input)
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn new_summary<T: Copy + Ord + Default, N>(input: &Vec<T>, _: &N) -> Option<Summary<T>>
where
    T: ToPrimitive,
{
    Summary::from(input)
}

fn main() {
    let mut b = Benchmark::default();
    b.add_generator(RandomVec::<i64>::new(1_000));
    b.add_pair(
        _benchmark_fn("old", old_summary),
        _benchmark_fn("new", new_summary),
    );

    cli::run(b, Default::default())
}
