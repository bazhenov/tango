use crate::{Benchmark, Generator, RunMode};
use clap::Parser;
use core::fmt;
use statrs::distribution::Normal;
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

    #[arg(short = 'v', long = "verbose", default_value_t = false)]
    verbose: bool,

    #[arg(long = "bench", default_value_t = true)]
    bench: bool,
}

pub fn run<P, O>(mut benchmark: Benchmark<P, O>, payloads: &mut dyn Generator<Output = P>) {
    let opts = Opts::parse();

    if opts.verbose {
        benchmark.add_reporter(VerboseReporter);
    } else {
        benchmark.add_reporter(ConsoleReporter::default());
    }

    match opts.subcommand {
        BenchMode::Pair {
            name,
            time,
            iterations,
            path_to_dump,
            bench: _,
        } => {
            benchmark.set_run_mode(determine_run_mode(time, iterations));
            benchmark.set_measurements_dir(path_to_dump.clone());
            let name = name.as_deref().unwrap_or("");
            benchmark.run_by_name(payloads, name);
        }
        BenchMode::Calibrate { bench: _ } => {
            benchmark.run_calibration();
        }
        BenchMode::List { bench: _ } => {
            for fn_name in benchmark.list_functions() {
                println!("{}", fn_name);
            }
        }
    }
}

fn determine_run_mode(time: Option<NonZeroU64>, iterations: Option<NonZeroUsize>) -> RunMode {
    let time = time.map(|t| RunMode::Time(Duration::from_millis(u64::from(t))));
    let iterations = iterations.map(RunMode::Iterations);
    time.or(iterations)
        .unwrap_or(RunMode::Time(Duration::from_millis(100)))
}

pub mod reporting {

    use crate::cli::{colorize, outliers_threshold, HumanTime};
    use crate::{Reporter, Summary};

    #[derive(Default)]
    pub(super) struct VerboseReporter;

    impl Reporter for VerboseReporter {
        fn on_complete(
            &mut self,
            base: (&str, Summary<i64>),
            candidate: (&str, Summary<i64>),
            measurements: &[i64],
        ) {
            const HR: &str = "----------------------------------";

            let (base_name, base) = base;
            let (candidate_name, candidate) = candidate;

            let n = measurements.len();
            let (min, max) =
                outliers_threshold(measurements.to_vec()).unwrap_or((i64::MIN, i64::MAX));

            let measurements = measurements
                .iter()
                .copied()
                .filter(|i| min < *i && *i < max)
                .collect::<Vec<_>>();
            let outliers_filtered = n - measurements.len();

            let diff_summary = Summary::from(measurements.as_slice());

            println!("{:12} {:>10} {:>10}", "", base_name, candidate_name);
            println!("{:12} {:>10} {:>10}", "n", base.n, candidate.n);
            println!(
                "{:12} {:>10} {:>10}",
                "min",
                HumanTime(base.min as f64),
                HumanTime(candidate.min as f64)
            );
            println!(
                "{:12} {:>10} {:>10}",
                "mean",
                HumanTime(base.mean),
                HumanTime(candidate.mean)
            );
            println!(
                "{:12} {:>10} {:>10}",
                "max",
                HumanTime(base.max as f64),
                HumanTime(candidate.max as f64)
            );
            println!(
                "{:12} {:>10} {:>10}",
                "std. dev.",
                HumanTime(base.variance.sqrt()),
                HumanTime(candidate.variance.sqrt())
            );
            println!();

            println!(
                "{:12} {:>10} {:>10}",
                "∆ mean",
                "",
                HumanTime(diff_summary.mean)
            );
            println!(
                "{:12} {:>10} {:>9.1}%",
                "∆ mean %",
                "",
                diff_summary.mean / base.mean * 100.
            );
            println!(
                "{:12} {:>10} {:>10}",
                "∆ std. dev.",
                "",
                HumanTime(diff_summary.variance.sqrt())
            );
            println!("{:12} {:>10} {:>10}", "outliers", "", outliers_filtered);

            let std_dev = diff_summary.variance.sqrt();
            let std_err = std_dev / (measurements.len() as f64).sqrt();
            let z_score = diff_summary.mean / std_err;

            let significant = z_score.abs() >= 2.6;
            if significant {
                println!(
                    "{:12} {:>10} {:>10}",
                    "CHANGE",
                    if diff_summary.mean > 0. { "FASTER" } else { "" },
                    if diff_summary.mean < 0. { "FASTER" } else { "" }
                );
            }
            println!("{}", HR);
        }
    }

    #[derive(Default)]
    pub(super) struct ConsoleReporter;

    impl Reporter for ConsoleReporter {
        fn on_complete(
            &mut self,
            base: (&str, Summary<i64>),
            candidate: (&str, Summary<i64>),
            measurements: &[i64],
        ) {
            let (base_name, base) = base;
            let (candidate_name, candidate) = candidate;

            let n = measurements.len() as f64;

            let diff = Summary::from(measurements);

            let std_dev = diff.variance.sqrt();
            let std_err = std_dev / n.sqrt();
            let z_score = diff.mean / std_err;

            let significant = z_score > 2.6;

            let speedup = (candidate.mean - base.mean) / base.mean * 100.;
            let candidate_faster = candidate.mean < base.mean;
            println!(
                "{:15} ... {:15} [ {:>8} ... {:>8} ]    {:5.1}%",
                base_name,
                colorize(candidate_name, significant, candidate_faster),
                HumanTime(base.mean),
                colorize(HumanTime(candidate.mean), significant, candidate_faster),
                colorize(speedup, significant, speedup < 0.),
            )
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn check_summary_statistics() {
            let stat = Summary::from(vec![1, 1, 2, 4].as_slice());
            assert_eq!(stat.min, 1);
            assert_eq!(stat.max, 4);
            assert_eq!(stat.variance, 2.);
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
    Reset,
}

impl Color {
    fn ascii_color_code(&self) -> &'static str {
        match self {
            Color::Red => "\x1B[31m",
            Color::Green => "\x1B[32m",
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

/// Running variance iterator
///
/// Provides a running (streaming variance) for a given iterator of observations.
/// Uses simple variance formula: `Var(X) = E[X^2] - E[X]^2`.
struct RunningVariance<T> {
    iter: T,
    sum: f64,
    sum_of_squares: f64,
    n: f64,
}

impl<T: Iterator<Item = i64>> Iterator for RunningVariance<T> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.iter.next()? as f64;
        self.sum += i;
        self.sum_of_squares += i.powi(2);
        self.n += 1.;

        Some((self.sum_of_squares / self.n) - (self.sum / self.n).powi(2))
    }
}

impl<I, T: Iterator<Item = I>> From<T> for RunningVariance<T> {
    fn from(value: T) -> Self {
        Self {
            iter: value,
            sum: 0.,
            sum_of_squares: 0.,
            n: 0.,
        }
    }
}

/// Outlier threshold detection
///
/// This functions detects optimal threshold for outlier filtering. Algorithm finds a threshold
/// that split the set of all observations `M` into two different subsets `S` and `O`. Each observation
/// is considered as a split point. Algorithm chooses split point in such way that it maximizes
/// the ration of `S` with this observation and without.
///
/// For example in a set of observations `[1, 2, 3, 100, 200, 300]` the target observation will be 100.
/// It is the observation including which will raise variance the most.
fn outliers_threshold(mut input: Vec<i64>) -> Option<(i64, i64)> {
    // TODO(bazhenov) sorting should be done by difference with median
    input.sort_by_key(|a| a.abs());
    let variance = RunningVariance::from(input.iter().copied());

    // Looking only 30% topmost values
    let mut outliers_cnt = input.len() * 30 / 100;
    let skip = input.len() - outliers_cnt;
    let mut candidate_outliers = input[skip..].iter().filter(|i| **i < 0).count();
    let value_and_variance = input.iter().copied().zip(variance).skip(skip);

    let mut prev_variance = 0.;
    for (value, var) in value_and_variance {
        if prev_variance > 0. && var / prev_variance > 1.2 {
            if let Some((min, max)) = binomial_interval_approximation(outliers_cnt, 0.5) {
                if candidate_outliers < min || candidate_outliers > max {
                    continue;
                }
            }
            return Some((-value.abs(), value.abs()));
        }
        prev_variance = var;
        outliers_cnt -= 1;
        if value < 0 {
            candidate_outliers -= 1;
        }
    }

    None
}

fn binomial_interval_approximation(n: usize, p: f64) -> Option<(usize, usize)> {
    use statrs::distribution::ContinuousCDF;
    let nf = n as f64;
    if nf * p < 10. || nf * (1. - p) < 10. {
        return None;
    }
    let mu = nf * p;
    let sigma = (nf * p * (1. - p)).sqrt();
    let distribution = Normal::new(mu, sigma).unwrap();
    let min = distribution.inverse_cdf(0.1).floor() as usize;
    let max = n - min;
    Some((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, RngCore, SeedableRng};

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

    struct RngIterator<T>(T);

    impl<T: RngCore> Iterator for RngIterator<T> {
        type Item = u32;

        fn next(&mut self) -> Option<Self::Item> {
            Some(self.0.next_u32())
        }
    }

    #[test]
    fn check_running_variance() {
        let input = [1i64, 2, 3, 4, 5, 6, 7];
        let variances = RunningVariance::from(input.into_iter());
        let expected = &[0., 0.25, 0.666, 1.25, 2.0, 2.916];

        for (value, expected_value) in variances.zip(expected.iter()) {
            assert!(
                (value - expected_value).abs() < 1e-3,
                "Expected close to: {}, given: {}",
                expected_value,
                value
            );
        }
    }

    #[test]
    fn check_running_variance_stress_test() {
        let rng = RngIterator(SmallRng::seed_from_u64(0)).map(|i| i as i64);
        let mut variances = RunningVariance::from(rng);

        assert!(variances.nth(10000000).unwrap() > 0.)
    }

    #[test]
    fn check_filter_outliers() {
        let input = vec![
            1i64, -2, 3, -4, 5, -6, 7, -8, 9, -10, //
            101, -102,
        ];

        let (min, max) = outliers_threshold(input).unwrap();
        assert!(min < 1, "Minimum is: {}", min);
        assert!(10 < max && max <= 101, "Maximum is: {}", max);
    }

    #[test]
    fn check_binomial_approximation() {
        assert_eq!(
            binomial_interval_approximation(10000000, 0.5),
            Some((4997973, 5002027))
        );
    }
}
