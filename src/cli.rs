use crate::{Benchmark, Generator, RunOpts};
use clap::Parser;
use core::fmt;
use std::{
    fmt::Display,
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
    time::Duration,
};

use self::reporting::{ConsoleReporter, VerboseReporter};

#[derive(Parser, Debug)]
enum BenchMode {
    Pair {
        name: Option<String>,

        #[arg(long = "bench", default_value_t = true)]
        bench: bool,

        #[arg(short = 'i', long = "iterations")]
        iterations: Option<NonZeroUsize>,

        #[arg(short = 't', long = "time")]
        time: Option<NonZeroU64>,

        /// write CSV dumps of all the measurements in a given location
        #[arg(short = 'd', long = "dump")]
        path_to_dump: Option<PathBuf>,

        /// disable outlier detection
        #[arg(short = 'o', long = "no-outliers")]
        skip_outlier_detection: bool,

        #[arg(short = 'v', long = "verbose", default_value_t = false)]
        verbose: bool,
    },
    Calibrate {
        #[arg(long = "bench", default_value_t = true)]
        bench: bool,
    },
    List {
        #[arg(long = "bench", default_value_t = true)]
        bench: bool,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Opts {
    #[command(subcommand)]
    subcommand: BenchMode,

    #[arg(long = "bench", default_value_t = true)]
    bench: bool,
}

pub fn run<P, O>(mut benchmark: Benchmark<P, O>, payloads: &mut [&mut dyn Generator<Output = P>]) {
    let opts = Opts::parse();

    match opts.subcommand {
        BenchMode::Pair {
            name,
            time,
            iterations,
            verbose,
            path_to_dump,
            skip_outlier_detection,
            bench: _,
        } => {
            if verbose {
                let mut reporter = VerboseReporter::default();
                reporter.set_skip_outlier_filtering(skip_outlier_detection);
                benchmark.add_reporter(reporter);
            } else {
                let mut reporter = ConsoleReporter::default();
                reporter.set_skip_outlier_filtering(skip_outlier_detection);
                benchmark.add_reporter(reporter);
            }

            let max_iterations = iterations.map(|i| i.into()).unwrap_or(1_000_000);
            let time = time.map(|i| i.into()).unwrap_or(100);

            let opts = RunOpts {
                name_filter: name,
                measurements_path: path_to_dump,
                max_iterations,
                max_duration: Duration::from_millis(time),
                outlier_detection_enabled: !skip_outlier_detection,
            };
            for generator in payloads {
                benchmark.run_by_name(*generator, &opts);
            }
        }
        BenchMode::Calibrate { bench: _ } => {
            benchmark.run_calibration(payloads[0]);
        }
        BenchMode::List { bench: _ } => {
            for fn_name in benchmark.list_functions() {
                println!("{}", fn_name);
            }
        }
    }
}

pub mod reporting {

    use crate::cli::{colorize, Color, Colored, HumanTime};
    use crate::{Reporter, RunResult};

    #[derive(Default)]
    pub(super) struct VerboseReporter {
        skip_outlier_filtering: bool,
    }

    impl VerboseReporter {
        pub fn set_skip_outlier_filtering(&mut self, flag: bool) {
            self.skip_outlier_filtering = flag
        }
    }

    impl Reporter for VerboseReporter {
        fn on_complete(&mut self, results: &RunResult) {
            let base = results.baseline;
            let candidate = results.candidate;

            let significant = results.significant;

            println!(
                "{} vs. {}  (n: {}, outliers: {})",
                Colored(&results.base_name, Color::Bold),
                Colored(&results.candidate_name, Color::Bold),
                results.diff.n,
                results.outliers
            );
            println!();

            println!(
                "    {:12}   {:>15} {:>15} {:>15}",
                "",
                Colored(&results.base_name, Color::Bold),
                Colored(&results.candidate_name, Color::Bold),
                Colored("∆", Color::Bold),
            );
            println!(
                "    {:12} ╭────────────────────────────────────────────────",
                ""
            );
            println!(
                "    {:12} │ {:>15} {:>15} {:>15}",
                "min",
                HumanTime(base.min as f64),
                HumanTime(candidate.min as f64),
                HumanTime((candidate.min - base.min) as f64)
            );
            println!(
                "    {:12} │ {:>15} {:>15} {:>15}  {:+3.1}{}",
                "mean",
                HumanTime(base.mean),
                HumanTime(candidate.mean),
                colorize(
                    HumanTime(results.diff.mean),
                    significant,
                    results.diff.mean < 0.
                ),
                colorize(
                    results.diff.mean / base.mean * 100.,
                    significant,
                    results.diff.mean < 0.
                ),
                colorize("%", significant, results.diff.mean < 0.)
            );
            println!(
                "    {:12} │ {:>15} {:>15} {:>15}",
                "max",
                HumanTime(base.max as f64),
                HumanTime(candidate.max as f64),
                HumanTime((candidate.max - base.max) as f64),
            );
            println!(
                "    {:12} │ {:>15} {:>15} {:>15}",
                "std. dev.",
                HumanTime(base.variance.sqrt()),
                HumanTime(candidate.variance.sqrt()),
                HumanTime(results.diff.variance.sqrt()),
            );
            println!();
        }
    }

    #[derive(Default)]
    pub(super) struct ConsoleReporter {
        skip_outlier_filtering: bool,
    }

    impl ConsoleReporter {
        pub fn set_skip_outlier_filtering(&mut self, flag: bool) {
            self.skip_outlier_filtering = flag
        }
    }

    impl Reporter for ConsoleReporter {
        fn on_start(&mut self, payloads_name: &str) {
            println!("{}", payloads_name);
        }

        fn on_complete(&mut self, results: &RunResult) {
            let base = results.baseline;
            let candidate = results.candidate;

            let significant = results.significant;

            let speedup = (candidate.mean - base.mean) / base.mean * 100.;
            let candidate_faster = candidate.mean < base.mean;
            println!(
                "  {:20} ... {:20} [ {:>8} ... {:>8} ]    {:>+5.1}%",
                results.base_name,
                colorize(&results.candidate_name, significant, candidate_faster),
                HumanTime(base.mean),
                colorize(HumanTime(candidate.mean), significant, candidate_faster),
                colorize(speedup, significant, speedup < 0.),
            )
        }
    }
}

fn colorize<T: Display>(value: T, do_paint: bool, indicator: bool) -> Colored<T> {
    if do_paint {
        let color = if indicator { Color::Green } else { Color::Red };
        Colored(value, color)
    } else {
        Colored(value, Color::Reset)
    }
}

enum Color {
    Red,
    Green,
    Bold,
    Reset,
}

impl Color {
    fn ascii_color_code(&self) -> &'static str {
        match self {
            Color::Red => "\x1B[31m",
            Color::Green => "\x1B[32m",
            Color::Bold => "\x1B[1m",
            Color::Reset => "\x1B[0m",
        }
    }
}

struct Colored<T>(T, Color);

impl<T: Display> Display for Colored<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.1.ascii_color_code())
            .and_then(|_| self.0.fmt(f))
            .and_then(|_| write!(f, "{}", Color::Reset.ascii_color_code()))
    }
}

struct HumanTime(f64);

impl fmt::Display for HumanTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const USEC: f64 = 1_000.;
        const MSEC: f64 = USEC * 1_000.;
        const SEC: f64 = MSEC * 1_000.;

        if self.0.abs() > SEC {
            f.pad(&format!("{:.1} s", self.0 / SEC))
        } else if self.0.abs() > MSEC {
            f.pad(&format!("{:.1} ms", self.0 / MSEC))
        } else if self.0.abs() > USEC {
            f.pad(&format!("{:.1} us", self.0 / USEC))
        } else {
            f.pad(&format!("{:.0} ns", self.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_human_time() {
        assert_eq!(format!("{}", HumanTime(0.1)), "0 ns");
        assert_eq!(format!("{:>5}", HumanTime(0.)), " 0 ns");

        assert_eq!(format!("{}", HumanTime(120.)), "120 ns");

        assert_eq!(format!("{}", HumanTime(1200.)), "1.2 us");

        assert_eq!(format!("{}", HumanTime(1200000.)), "1.2 ms");

        assert_eq!(format!("{}", HumanTime(1200000000.)), "1.2 s");

        assert_eq!(format!("{}", HumanTime(-1200000.)), "-1.2 ms");
    }
}
