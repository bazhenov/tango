#![cfg_attr(feature = "align", feature(fn_align))]

use rand::{rngs::SmallRng, Rng, SeedableRng};
use tango_bench::{
    benchmark, benchmark_fn, benchmark_fn_with_setup, cli::run, prevent_shared_function_deletion,
    Benchmark, BenchmarkFn, Generator, MeasurementSettings, StaticValue,
};
use test_funcs::{factorial, str_count, str_count_rev, str_std, str_take, sum, RandomString};

mod test_funcs;

struct RandomVec(SmallRng, usize);

impl Generator for RandomVec {
    type Haystack = Vec<u32>;
    type Needle = ();

    fn next_haystack(&mut self) -> Self::Haystack {
        let RandomVec(rng, size) = self;
        let mut v = vec![0; *size];
        rng.fill(&mut v[..]);
        v
    }

    fn name(&self) -> String {
        format!("RandomVec<{}>", self.1)
    }

    fn next_needle(&mut self, _haystack: &Self::Haystack) -> Self::Needle {}
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn sort_unstable<T: Ord + Copy, N>(mut input: Vec<T>, _: &N) -> T {
    input.sort_unstable();
    input[input.len() / 2]
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn sort_stable<T: Ord + Copy, N>(mut input: Vec<T>, _: &N) -> T {
    input.sort();
    input[input.len() / 2]
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(clippy::ptr_arg)]
fn copy_and_sort_stable<T: Ord + Copy, N>(input: &Vec<T>, _: &N) -> T {
    let mut input = input.clone();
    input.sort();
    input[input.len() / 2]
}

// trait MeasureTarget {
//     fn measure(&mut self, iterations: usize);
// }

// struct Pair<H: 'static, N: 'static, O: 'static> {
//     f: &'static dyn BenchmarkFn<H, N, O>,
//     g: &'static dyn Generator<Haystack = H, Needle = N>,
// }

#[used]
#[no_mangle]
static BENCHMARK: &[u64] = &[0, 2, 4, 8];

fn main() {
    // prevent_shared_function_deletion!();

    let settings = MeasurementSettings::default();

    let mut num_benchmark = Benchmark::default();
    num_benchmark.add_generator(StaticValue((), ()));

    #[cfg(feature = "aa_test")]
    {
        benchmark.add_pair(
            benchmark_fn("sum_5000", |_, _| sum(5000)),
            benchmark_fn("sum_5000", |_, _| sum(5000)),
        );

        benchmark.add_pair(
            benchmark_fn("factorial_500", |_, _| factorial(500)),
            benchmark_fn("factorial_500", |_, _| factorial(500)),
        );
    }

    num_benchmark.add_pair(
        benchmark_fn("sum_5000", |_, _| sum(5000)),
        benchmark_fn("sum_4950", |_, _| sum(4950)),
    );

    num_benchmark.add_pair(
        benchmark_fn("factorial_500", |_, _| factorial(500)),
        benchmark_fn("factorial_495", |_, _| factorial(495)),
    );

    let mut str_benchmark = Benchmark::default();
    str_benchmark.add_generator(RandomString::new().unwrap());

    str_benchmark.add_pair(
        benchmark_fn("str_std", str_std),
        benchmark_fn("str_count", str_count),
    );
    str_benchmark.add_pair(
        benchmark_fn("str_count", str_count),
        benchmark_fn("str_count_rev", str_count_rev),
    );
    str_benchmark.add_pair(
        benchmark_fn("str_5000", |h, n| str_take(5000, h, n)),
        benchmark_fn("str_4950", |h, n| str_take(4950, h, n)),
    );

    let mut sort_benchmark = Benchmark::default();

    for size in [100, 1_000, 10_000, 100_000] {
        sort_benchmark.add_generator(RandomVec(SmallRng::seed_from_u64(42), size));
    }

    sort_benchmark.add_pair(
        benchmark_fn_with_setup("stable", sort_stable, Clone::clone),
        benchmark_fn_with_setup("unstable", sort_unstable, Clone::clone),
    );

    sort_benchmark.add_pair(
        benchmark_fn_with_setup("stable", sort_stable, Clone::clone),
        benchmark_fn("stable_clone_sort", copy_and_sort_stable),
    );

    run(str_benchmark, settings);
    run(sort_benchmark, settings);
    run(num_benchmark, settings);
}
