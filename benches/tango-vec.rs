use rand::{rngs::SmallRng, Rng, SeedableRng};
use rust_pairwise_testing::{
    benchmark_fn, benchmark_fn_with_setup, cli::run, Benchmark, Generator,
};
use std::num::NonZeroUsize;

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

fn sort_unstable(input: &Vec<u32>) -> usize {
    let mut copy = input.clone();
    copy.sort_unstable();
    copy.len()
}

fn sort_stable(mut input: Vec<u32>) -> usize {
    input.sort();
    input.len()
}

fn copy_and_sort_stable(input: &Vec<u32>) -> usize {
    let mut input = input.clone();
    input.sort();
    input.len()
}

fn main() {
    let mut benchmark = Benchmark::new(RandomVec(
        SmallRng::seed_from_u64(42),
        NonZeroUsize::new(100).unwrap(),
    ));

    benchmark.add_pair(
        "stable-unstable",
        benchmark_fn_with_setup(sort_stable, Clone::clone),
        benchmark_fn(sort_unstable),
    );
    benchmark.add_pair(
        "stable-copy_stable",
        benchmark_fn_with_setup(sort_stable, Clone::clone),
        benchmark_fn(copy_and_sort_stable),
    );

    run(benchmark)
}
