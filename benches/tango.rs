// #![feature(fn_align)]
use clap::Parser;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rust_pairwise_testing::{
    benchmark_fn, benchmark_fn_with_setup, reporting::ConsoleReporter, Benchmark, Generator,
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

#[derive(Parser, Debug)]
enum RunMode {
    Pair(PairOpts),
    Calibration(CalibrationOpts),
}

#[derive(Parser, Debug)]
struct PairOpts {
    functions: Vec<String>,

    #[arg(long = "bench", default_value_t = true)]
    bench: bool,
}

#[derive(Parser, Debug)]
struct CalibrationOpts {
    #[arg(long = "bench", default_value_t = true)]
    bench: bool,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Opts {
    #[command(subcommand)]
    subcommand: RunMode,

    #[arg(long = "bench", default_value_t = true)]
    bench: bool,
}

fn main() {
    let opts = Opts::parse();

    let mut benchmark = Benchmark::new(RandomVec(
        SmallRng::seed_from_u64(42),
        NonZeroUsize::new(100).unwrap(),
    ));

    benchmark.set_iterations(5000);

    benchmark.add_function("stable", benchmark_fn_with_setup(sort_stable, Clone::clone));
    benchmark.add_function("copy_stable", benchmark_fn(copy_and_sort_stable));
    benchmark.add_function("unstable", benchmark_fn(sort_unstable));

    let mut reporter = ConsoleReporter::default();

    match opts.subcommand {
        RunMode::Pair(opts) => {
            assert!(opts.functions.len() >= 2);
            let baseline = &opts.functions[0];
            for candidate in &opts.functions[1..] {
                benchmark.run_pair(baseline, candidate, &mut reporter);
            }
        }
        RunMode::Calibration(_) => {
            benchmark.run_calibration(&mut reporter);
        }
    }

    // benchmark.run_pair("stable", "unstable", &mut reporter);

    // benchmark.run_pair("stable", "unstable", &mut reporter);
    // benchmark.run_pair("stable", "copy_stable", &mut reporter);
}
