#![cfg_attr(feature = "align", feature(fn_align))]

use std::{cell::RefCell, rc::Rc};

use num_traits::ToPrimitive;
use tango_bench::{
    benchmark_fn, benchmarks, cli, BenchmarkMatrix, GenAndFunc, IntoBenchmarks, MeasureTarget,
    StaticValue, Summary, _benchmark_fn,
};
use test_funcs::RandomVec;

mod test_funcs;

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn create_summary<T: Copy + Ord + Default, N>(input: &Vec<T>, _: &N) -> Option<Summary<T>>
where
    T: ToPrimitive,
{
    Summary::from(input)
}

fn summary_benchmarks() -> impl IntoBenchmarks {
    let generator = RandomVec::<i64>::new(1_000);
    BenchmarkMatrix::new(generator).add_function("summary", create_summary)
}

fn empty_benchmarks() -> impl IntoBenchmarks {
    [benchmark_fn("measure_empty_function", || {
        benchmark_fn("_", || 42).measure(1000)
    })]
}

fn generator_empty_benchmarks() -> impl IntoBenchmarks {
    let mut generator = GenAndFunc::new(
        _benchmark_fn("_", |_, needle| *needle),
        StaticValue(0usize, 0usize),
    );

    // warming up
    generator.measure(1000);
    let rator = StaticValue(Rc::new(RefCell::new(generator)), ());
    BenchmarkMatrix::new(rator).add_function("measure_generator_function", |rator, _| {
        rator.borrow_mut().measure(1000)
    })
}

benchmarks!(
    empty_benchmarks(),
    generator_empty_benchmarks(),
    summary_benchmarks()
);

fn main() {
    cli::run(Default::default())
}
