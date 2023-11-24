#![cfg_attr(feature = "align", feature(fn_align))]

use rand::{rngs::SmallRng, Rng, SeedableRng};
use tango_bench::{
    _benchmark_fn, benchmark_fn, benchmark_fn_with_setup, cli::run, Benchmark, Generator,
    GeneratorBenchmarks, IntoBenchmarks, MeasureTarget, MeasurementSettings, StaticValue,
};
use test_funcs::{factorial, str_count, str_count_rev, str_std, str_take, sum, RandomString};

mod test_funcs;

#[derive(Clone)]
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

pub fn num_benchmarks() -> impl IntoBenchmarks {
    vec![
        benchmark_fn("sum_5000", || sum(5000)),
        benchmark_fn("sum_4950", || sum(4950)),
        benchmark_fn("factorial_500", || factorial(500)),
        benchmark_fn("factorial_495", || factorial(495)),
    ]
}

pub fn str_benchmarks() -> impl IntoBenchmarks {
    let generator = RandomString::new().unwrap();
    let mut benchmarks = GeneratorBenchmarks::with_generator(generator);

    benchmarks
        .add("str_std", str_std)
        .add("str_count", str_count)
        .add("str_count_rev", str_count_rev)
        .add("str_5000", |h, n| str_take(5000, h, n))
        .add("str_4950", |h, n| str_take(4950, h, n));

    benchmarks
}

pub fn vec_benchmarks() -> impl IntoBenchmarks {
    let mut benchmarks =
        GeneratorBenchmarks::with_generators([100, 1_000, 10_000, 100_000], |size| {
            RandomVec(SmallRng::seed_from_u64(42), size)
        });

    benchmarks.add("stable_clone_sort", copy_and_sort_stable);

    benchmarks
}

#[no_mangle]
pub fn create_benchmarks() -> Vec<Box<dyn MeasureTarget>> {
    let mut benchmarks = vec![];
    benchmarks.extend(num_benchmarks().into_benchmarks());
    benchmarks.extend(str_benchmarks().into_benchmarks());
    benchmarks.extend(vec_benchmarks().into_benchmarks());
    benchmarks
}

fn main() {
    // prevent_shared_function_deletion!();

    let settings = MeasurementSettings::default();

    let mut num_benchmark = Benchmark::default();
    num_benchmark.add_generator(StaticValue((), ()));

    #[cfg(feature = "aa_test")]
    {
        num_benchmark.add_pair(
            _benchmark_fn("sum_5000", |_, _| sum(5000)),
            _benchmark_fn("sum_5000", |_, _| sum(5000)),
        );

        num_benchmark.add_pair(
            _benchmark_fn("factorial_500", |_, _| factorial(500)),
            _benchmark_fn("factorial_500", |_, _| factorial(500)),
        );
    }

    num_benchmark.add_pair(
        _benchmark_fn("sum_5000", |_, _| sum(5000)),
        _benchmark_fn("sum_4950", |_, _| sum(4950)),
    );

    num_benchmark.add_pair(
        _benchmark_fn("factorial_500", |_, _| factorial(500)),
        _benchmark_fn("factorial_495", |_, _| factorial(495)),
    );

    let mut str_benchmark = Benchmark::default();
    str_benchmark.add_generator(RandomString::new().unwrap());

    #[cfg(feature = "aa_test")]
    {
        str_benchmark.add_pair(
            _benchmark_fn("str_count_rev", str_count_rev),
            _benchmark_fn("str_count_rev", str_count_rev),
        );

        str_benchmark.add_pair(
            _benchmark_fn("str_count", str_count),
            _benchmark_fn("str_count", str_count),
        );
    }

    str_benchmark.add_pair(
        _benchmark_fn("str_std", str_std),
        _benchmark_fn("str_count", str_count),
    );
    str_benchmark.add_pair(
        _benchmark_fn("str_count", str_count),
        _benchmark_fn("str_count_rev", str_count_rev),
    );
    str_benchmark.add_pair(
        _benchmark_fn("str_5000", |h, n| str_take(5000, h, n)),
        _benchmark_fn("str_4950", |h, n| str_take(4950, h, n)),
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
        _benchmark_fn("stable_clone_sort", copy_and_sort_stable),
    );

    run(str_benchmark, settings);
    // run(sort_benchmark, settings);
    // run(num_benchmark, settings);
}
