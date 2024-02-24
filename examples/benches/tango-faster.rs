#![cfg_attr(feature = "align", feature(fn_align))]

use std::rc::Rc;

use crate::test_funcs::{factorial, sum};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tango_bench::{
    generators::RandomVec, new_api::new_bench, tango_benchmarks, tango_main, BenchmarkMatrix,
    IntoBenchmarks,
};
use test_funcs::{
    build_char_indicies, sort_unstable, str_count, str_count_new, str_take, RandomSubstring,
    INPUT_TEXT,
};

mod test_funcs;

fn str_benchmarks() -> impl IntoBenchmarks {
    BenchmarkMatrix::new(RandomSubstring::new())
        .add_function("str_length", str_count)
        .add_function("str_length_limit", |h, n| str_take(4950, h, n))
}

fn num_benchmarks() -> impl IntoBenchmarks {
    [
        new_bench("sum", |_| || sum(4950)),
        new_bench("factorial", |_| || factorial(495)),
    ]
}

fn vec_benchmarks() -> impl IntoBenchmarks {
    BenchmarkMatrix::with_params([100, 1_000, 10_000, 100_000], RandomVec::<u64>::new)
        .add_function("sort", sort_unstable)
}

fn new_benchmarks() -> impl IntoBenchmarks {
    let mut benches = vec![];
    let input_string = INPUT_TEXT;
    let indicies = Rc::new(build_char_indicies(input_string));
    for length in [5, 500, 50_000] {
        let indicies = Rc::clone(&indicies);
        let bench = new_bench(format!("str_length/length<{}>", length), move |p| {
            let mut rng = SmallRng::seed_from_u64(p.seed);
            let indicies = Rc::clone(&indicies);
            move || {
                let start = rng.gen_range(0..indicies.len() - length);
                let range = indicies[start]..indicies[start + length];
                str_count_new(&input_string[range])
            }
        });
        benches.push(bench);
    }

    benches
}

tango_benchmarks!(
    str_benchmarks(),
    num_benchmarks(),
    vec_benchmarks(),
    new_benchmarks()
);
tango_main!();
