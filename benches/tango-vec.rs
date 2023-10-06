use rand::{rngs::SmallRng, Rng, SeedableRng};
use rust_pairwise_testing::{
    benchmark_fn, benchmark_fn_with_setup, cli::run, Benchmark, Generator,
};
use std::{hint::black_box, num::NonZeroUsize};

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
fn sort_unstable(input: &Vec<u32>) -> usize {
    let mut copy = input.clone();
    copy.sort_unstable();
    copy.len()
}

//#[repr(align(32))]
fn sort_stable(mut input: Vec<u32>) -> usize {
    input.sort();
    input.len()
}

//#[repr(align(32))]
fn copy_and_sort_stable(input: &Vec<u32>) -> usize {
    let mut input = input.clone();
    input.sort();
    input.len()
}

fn sum_49250(_: &Vec<u32>) -> usize {
    sum(49250)
}

fn sum_50000(_: &Vec<u32>) -> usize {
    sum(50000)
}

fn sum(n: usize) -> usize {
    let mut sum = 0;
    for i in 0..black_box(n) {
        sum += black_box(i);
    }
    sum
}

fn main() {
    let mut benchmark = Benchmark::new(RandomVec(
        SmallRng::seed_from_u64(42),
        NonZeroUsize::new(100).unwrap(),
    ));

    benchmark.set_iterations(10000);

    benchmark.add_function("stable", benchmark_fn_with_setup(sort_stable, Clone::clone));
    benchmark.add_function("copy_stable", benchmark_fn(copy_and_sort_stable));
    benchmark.add_function("unstable", benchmark_fn(sort_unstable));

    benchmark.add_function("sum_50000", benchmark_fn(sum_50000));
    benchmark.add_function("sum_49250", benchmark_fn(sum_49250));

    run(benchmark)
}
