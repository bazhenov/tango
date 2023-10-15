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

fn sort_unstable(mut input: Vec<u32>) -> usize {
    input.sort_unstable();
    input.len()
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
    let mut payloads = RandomVec(SmallRng::seed_from_u64(42), NonZeroUsize::new(100).unwrap());
    let mut benchmark = Benchmark::new();

    benchmark.add_pair(
        benchmark_fn_with_setup("stable", sort_stable, Clone::clone),
        benchmark_fn_with_setup("unstable", sort_unstable, Clone::clone),
    );
    benchmark.add_pair(
        benchmark_fn_with_setup("stable", sort_stable, Clone::clone),
        benchmark_fn("stable_clone_sort", copy_and_sort_stable),
    );

    run(benchmark, &mut payloads)
}
