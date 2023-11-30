#![cfg_attr(feature = "align", feature(fn_align))]

use std::{cell::RefCell, rc::Rc};

use num_traits::ToPrimitive;
use tango_bench::{
    benchmark_fn, benchmarks, cli, BenchmarkMatrix, GenFunc, Generator, IntoBenchmarks,
    MeasureTarget, Summary,
};
use test_funcs::RandomVec;

mod test_funcs;

#[derive(Clone)]
struct StaticValue<H, N>(
    /// Haystack value
    pub H,
    /// Needle value
    pub N,
);

impl<H: Clone, N: Copy> Generator for StaticValue<H, N> {
    type Haystack = H;
    type Needle = N;

    fn next_haystack(&mut self) -> Self::Haystack {
        self.0.clone()
    }

    fn next_needle(&mut self, _: &Self::Haystack) -> Self::Needle {
        self.1
    }

    fn name(&self) -> &str {
        "StaticValue"
    }
}

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
    let generator = Rc::new(RefCell::new(StaticValue(0usize, 0usize)));
    let f = |_: &usize, needle: &usize| *needle;
    let mut target = GenFunc::new("_", Rc::new(RefCell::new(f)), generator);

    // warming up
    target.measure(1000);
    let rator = StaticValue(Rc::new(RefCell::new(target)), ());
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
