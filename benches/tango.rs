// #![feature(fn_align)]

use rand::{rngs::SmallRng, Rng, SeedableRng};
use rust_pairwise_testing::{reporting::ConsoleReporter, Benchmark, Func, Generator, SetupFunc};
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
fn copy_and_sort_stable(mut input: &Vec<u32>) -> usize {
    let mut input = input.clone();
    input.sort();
    input.len()
}

fn clone<T: Clone>(obj: &T) -> T {
    obj.clone()
}

fn main() {
    let a = 0u8;
    println!("Stack address: {:p}", &a);
    // println!(
    //     "sort_stable!: {:p}",
    //     sort_stable as fn(&Vec<u32>) -> Vec<u32>
    // );
    // println!(
    //     "sort_unstable!: {:p}",
    //     sort_unstable as fn(&Vec<u32>) -> Vec<u32>
    // );
    let mut benchmark = Benchmark::new(RandomVec(
        SmallRng::seed_from_u64(42),
        NonZeroUsize::new(100).unwrap(),
    ));

    benchmark.set_iterations(1000);

    benchmark.add_function(
        "stable",
        SetupFunc {
            setup: clone,
            func: sort_stable,
        },
    );
    benchmark.add_function(
        "copy_stable",
        Func {
            func: copy_and_sort_stable,
        },
    );
    benchmark.add_function(
        "unstable",
        Func {
            func: sort_unstable,
        },
    );

    let mut reporter = ConsoleReporter::default();

    // benchmark.run_pair("stable", "unstable", &mut reporter);

    benchmark.run_pair("stable", "unstable", &mut reporter);
    benchmark.run_pair("stable", "copy_stable", &mut reporter);
    benchmark.run_calibration(&mut reporter);
}
