use rand::{rngs::SmallRng, Rng};
use rust_pairwise_testing::{benchmark_fn, cli::run, Benchmark, Generator};
use std::num::NonZeroUsize;
use test_funcs::{std, std_4925, std_5000, std_count, std_count_rev, RandomStringGenerator};
mod test_funcs;

struct RandomVec(SmallRng, NonZeroUsize);

impl Generator for RandomVec {
    type Output = Vec<u32>;

    fn next_payload(&mut self) -> Self::Output {
        let RandomVec(rng, size) = self;
        let mut v = vec![0; (*size).into()];
        rng.fill(&mut v[..]);
        v
    }
}

//#[repr(align(32))]
pub fn sort_unstable(input: &Vec<u32>) -> usize {
    let mut copy = input.clone();
    copy.sort_unstable();
    copy.len()
}

//#[repr(align(32))]
pub fn sort_stable(mut input: Vec<u32>) -> usize {
    input.sort();
    input.len()
}

//#[repr(align(32))]
pub fn copy_and_sort_stable(input: &Vec<u32>) -> usize {
    let mut input = input.clone();
    input.sort();
    input.len()
}

fn main() {
    let mut benchmark = Benchmark::new(RandomStringGenerator::new().unwrap());

    benchmark.set_iterations(10000);

    benchmark.add_function("std", benchmark_fn(std));
    benchmark.add_function("std_count", benchmark_fn(std_count));
    benchmark.add_function("std_count_rev", benchmark_fn(std_count_rev));
    benchmark.add_function("std_4925", benchmark_fn(std_4925));
    benchmark.add_function("std_5000", benchmark_fn(std_5000));

    run(benchmark)
}
