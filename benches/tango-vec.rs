use rand::{rngs::SmallRng, Rng, SeedableRng};
use rust_pairwise_testing::{
    benchmark_fn, benchmark_fn_with_setup, cli::run, Benchmark, Generator,
};

struct RandomVec(SmallRng, usize);

impl Generator for RandomVec {
    type Output = Vec<u32>;

    fn next_payload(&mut self) -> Self::Output {
        let RandomVec(rng, size) = self;
        let mut v = vec![0; *size];
        rng.fill(&mut v[..]);
        v
    }

    fn name(&self) -> String {
        format!("RandomVec<{}>", self.1)
    }
}

fn sort_unstable<T: Ord + Copy>(mut input: Vec<T>) -> T {
    input.sort_unstable();
    input[input.len() / 2]
}

fn sort_stable<T: Ord + Copy>(mut input: Vec<T>) -> T {
    input.sort();
    input[input.len() / 2]
}

fn copy_and_sort_stable<T: Ord + Copy>(input: &Vec<T>) -> T {
    let mut input = input.clone();
    input.sort();
    input[input.len() / 2]
}

fn main() {
    let mut benchmark = Benchmark::new();

    benchmark.add_pair(
        benchmark_fn_with_setup("stable", sort_stable, Clone::clone),
        benchmark_fn_with_setup("unstable", sort_unstable, Clone::clone),
    );

    benchmark.add_pair(
        benchmark_fn_with_setup("stable", sort_stable, Clone::clone),
        benchmark_fn("stable_clone_sort", copy_and_sort_stable),
    );

    run(
        benchmark,
        &mut [
            &mut RandomVec(SmallRng::seed_from_u64(42), 100),
            &mut RandomVec(SmallRng::seed_from_u64(42), 1000),
            &mut RandomVec(SmallRng::seed_from_u64(42), 10000),
            &mut RandomVec(SmallRng::seed_from_u64(42), 100000),
        ],
    )
}
