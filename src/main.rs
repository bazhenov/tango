#![feature(fn_align)]

mod test_funcs;

use rust_pairwise_testing::{reporting::ConsoleReporter, Benchmark, RandomStringGenerator};
use std::io;
use test_funcs::{std, std_4925, std_5000, std_5000_dupl, std_count, std_count_rev};

fn main() -> io::Result<()> {
    let mut benchmark = Benchmark::new(RandomStringGenerator::new()?);
    benchmark.set_iterations(1000);

    benchmark.add_function("std", std);
    benchmark.add_function("std_count", std_count);
    benchmark.add_function("std_count_rev", std_count_rev);
    benchmark.add_function("std_5000", std_5000);
    benchmark.add_function("std_5000_dupl", std_5000_dupl);
    benchmark.add_function("std_4925", std_4925);

    let mut reporter = ConsoleReporter::default();
    // reporter.set_write_data(false);

    // benchmark.run_pair("std", "std", &mut reporter);

    // benchmark.run_all_against("std", &mut reporter);

    // benchmark.run_pair("std", "std_count", &mut reporter);

    benchmark.run_pair("std_count", "std_count_rev", &mut reporter);
    benchmark.run_pair("std_5000", "std_4925", &mut reporter);
    benchmark.run_pair("std_5000_dupl", "std_4925", &mut reporter);
    benchmark.run_pair("std_5000", "std_5000_dupl", &mut reporter);

    // benchmark.run_all_against("std", &mut reporter);
    benchmark.run_calibration(&mut reporter);

    Ok(())
}
