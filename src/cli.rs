use crate::{Benchmark, Reporter, RunMode};
use clap::Parser;
use core::fmt;
use std::{
    num::{NonZeroU64, NonZeroUsize},
    time::Duration,
};

use self::reporting::{NewConsoleReporter, VerboseReporter};

#[derive(Parser, Debug)]
enum BenchMode {
    Pair {
        baseline: String,
        candidates: Vec<String>,

        #[arg(long = "bench", default_value_t = true)]
        bench: bool,

        #[arg(short = 'i', long = "iterations")]
        iterations: Option<NonZeroUsize>,

        #[arg(short = 't', long = "time")]
        time: Option<NonZeroU64>,
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

pub fn run<P, O>(mut benchmark: Benchmark<P, O>) {
    let opts = Opts::parse();

    let mut console_reporter = NewConsoleReporter::default();
    let mut verbose_reporter = VerboseReporter::default();

    let reporter: &mut dyn Reporter = match opts.verbose {
        true => &mut verbose_reporter,
        false => &mut console_reporter,
    };

    match opts.subcommand {
        BenchMode::Pair {
            candidates,
            baseline,
            time,
            iterations,
            ..
        } => {
            benchmark.set_run_mode(determine_run_mode(time, iterations));
            for candidate in &candidates {
                benchmark.run_pair(&baseline, candidate, reporter);
            }
        }
        BenchMode::Calibrate { .. } => {
            benchmark.run_calibration(reporter);
        }
        BenchMode::List { .. } => {
            for fn_name in benchmark.list_functions() {
                println!("{}", fn_name);
            }
        }
    }
}

fn determine_run_mode(time: Option<NonZeroU64>, iterations: Option<NonZeroUsize>) -> RunMode {
    let time = time.map(|t| RunMode::Time(Duration::from_millis(u64::from(t))));
    let iterations = iterations.map(|i| RunMode::Iterations(i));
    time.or(iterations)
        .unwrap_or(RunMode::Time(Duration::from_millis(100)))
}

pub mod reporting {
    use crate::cli::{outliers_threshold, HumanTime};
    use crate::Reporter;
    use std::io::Write;
    use std::iter::Sum;
    use std::{fs::File, io::BufWriter};

    struct Summary<T> {
        n: usize,
        min: T,
        max: T,
        mean: f64,
        variance: f64,
    }

    // TODO(bazhenov) not need to use IntoIterator here, slice is enough
    impl<T: Ord + Copy + Sum, I: IntoIterator<Item = T>> From<I> for Summary<T>
    where
        i64: From<T>,
    {
        fn from(iter: I) -> Self {
            let values = iter.into_iter().collect::<Vec<_>>();
            let n = values.len();
            let min = *values.iter().min().unwrap();
            let max = *values.iter().max().unwrap();
            let sum = values.iter().copied().sum::<T>();

            let mean = i64::from(sum) as f64 / n as f64;

            let variance = values
                .iter()
                .map(|i| (i64::from(*i) as f64 - mean).powi(2))
                .sum::<f64>()
                / (n - 1) as f64;

            Self {
                n,
                min,
                max,
                mean,
                variance,
            }
        }
    }

    #[derive(Default)]
    pub struct VerboseReporter;

    impl Reporter for VerboseReporter {
        fn on_complete(&mut self, baseline: &str, candidate: &str, measurements: &[(u64, u64)]) {
            const HR: &str = "----------------------------------";
            let n = measurements.len();
            let diff = measurements
                .iter()
                .copied()
                .map(|(b, c)| c as i64 - b as i64)
                .collect::<Vec<_>>();
            let (min, max) = outliers_threshold(diff.clone()).unwrap_or((i64::MIN, i64::MAX));

            let (diff, measurements): (Vec<_>, Vec<_>) = diff
                .into_iter()
                .zip(measurements)
                .filter(|(diff, _)| min < *diff && *diff < max)
                .unzip();
            let outliers_filtered = n - measurements.len();

            let (base, cand): (Vec<_>, Vec<_>) = measurements
                .iter()
                .copied()
                .map(|(b, c)| (b as i64, c as i64))
                .unzip();

            let base_summary = Summary::from(base);
            let cand_summary = Summary::from(cand);
            let diff_summary = Summary::from(diff);

            println!("{:12} {:>10} {:>10}", "", baseline, candidate);
            println!("{:12} {:>10} {:>10}", "n", base_summary.n, cand_summary.n);
            println!(
                "{:12} {:>10} {:>10}",
                "min",
                HumanTime(base_summary.min as f64),
                HumanTime(cand_summary.min as f64)
            );
            println!(
                "{:12} {:>10} {:>10}",
                "max",
                HumanTime(base_summary.max as f64),
                HumanTime(cand_summary.max as f64)
            );
            println!(
                "{:12} {:>10} {:>10}",
                "mean",
                HumanTime(base_summary.mean),
                HumanTime(cand_summary.mean)
            );
            println!(
                "{:12} {:>10} {:>10}",
                "std. dev.",
                HumanTime(base_summary.variance.sqrt()),
                HumanTime(cand_summary.variance.sqrt())
            );
            println!();

            println!(
                "{:12} {:>10} {:>10}",
                "∆ mean",
                "",
                HumanTime(diff_summary.mean)
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
    pub struct NewConsoleReporter {
        header_printed: bool,
    }

    impl Reporter for NewConsoleReporter {
        fn on_complete(&mut self, baseline_name: &str, candidate_name: &str, input: &[(u64, u64)]) {
            const HR: &str = "--------------------------------------------------";

            if !self.header_printed {
                println!("{:>20} {:>9} {:>9} {:>9}", "name", "iters.", "min", "mean");
                println!("{}", HR);
                self.header_printed = true;
            }

            let base = input
                .iter()
                .map(|(base, _)| *base as i64)
                .collect::<Vec<_>>();
            let candidate = input
                .iter()
                .map(|(_, candidate)| *candidate as i64)
                .collect::<Vec<_>>();

            let base_min = *base.iter().min().unwrap();
            let candidate_min = *candidate.iter().min().unwrap();

            let n = base.len() as f64;

            let base_mean = base.iter().sum::<i64>() as f64 / n;
            let candidate_mean = candidate.iter().sum::<i64>() as f64 / n;
            let diff = input
                .iter()
                .map(|(base, candidate)| *candidate as i64 - *base as i64)
                .collect::<Vec<i64>>();

            let mean_of_diff = diff.iter().sum::<i64>() as f64 / n;
            let variance = diff
                .iter()
                .map(|i| (*i as f64 - mean_of_diff).powi(2))
                .sum::<f64>()
                / (n - 1.);
            let std_dev = variance.sqrt();
            let std_err = std_dev / n.sqrt();
            let z_score = mean_of_diff / std_err;

            println!(
                "B: {:17} {:9} {:>9} {:>9}",
                baseline_name,
                n,
                HumanTime(base_min as f64),
                HumanTime(base_mean as f64),
            );

            println!(
                "C: {:17} {:9} {:>9} {:>9}",
                candidate_name,
                "",
                HumanTime(candidate_min as f64),
                HumanTime(candidate_mean as f64),
            );

            let diff_min = candidate_min - base_min;
            println!(
                "   {:17} {:9} {:>9} {:>9}",
                "diff.",
                "",
                HumanTime(diff_min as f64),
                HumanTime(mean_of_diff as f64),
            );

            let significant = z_score.abs() >= 2.6;
            println!(
                "   {:17} {:9} {:>8.1}% {:>8.1}%   {}",
                "%",
                "",
                diff_min as f64 / base_min as f64 * 100.,
                (mean_of_diff / base_mean * 100.),
                match significant {
                    true if mean_of_diff > 0. => "C is slower",
                    true if mean_of_diff < 0. => "C is faster",
                    _ => "",
                }
            );

            println!("{}", HR);
        }
    }

    #[derive(Default)]
    pub struct ConsoleReporter {
        header_printed: bool,
        write_data: bool,
    }

    impl ConsoleReporter {
        pub fn set_write_data(&mut self, write_data: bool) {
            self.write_data = write_data;
        }
    }

    impl Reporter for ConsoleReporter {
        fn on_complete(&mut self, baseline_name: &str, candidate_name: &str, input: &[(u64, u64)]) {
            let name = format!("{} / {}", baseline_name, candidate_name);
            let base = input
                .iter()
                .map(|(base, _)| *base as i64)
                .collect::<Vec<_>>();
            let candidate = input
                .iter()
                .map(|(_, candidate)| *candidate as i64)
                .collect::<Vec<_>>();

            let base_min = *base.iter().min().unwrap();
            let candidate_min = *candidate.iter().min().unwrap();

            let base_max = *base.iter().max().unwrap();
            let candidate_max = *candidate.iter().max().unwrap();

            let n = base.len() as f64;

            let base_mean = base.iter().sum::<i64>() as f64 / n;
            let candidate_mean = candidate.iter().sum::<i64>() as f64 / n;
            let mut diff = input
                .iter()
                .map(|(base, candidate)| *candidate as i64 - *base as i64)
                .collect::<Vec<i64>>();

            let filtered = mask_symmetric_outliers(&mut diff);

            let diff_mean = diff.iter().sum::<i64>() as f64 / n;
            let variance = diff
                .iter()
                .map(|i| (*i as f64 - diff_mean).powi(2))
                .sum::<f64>()
                / (n - 1.);
            let std_dev = variance.sqrt();
            let std_err = std_dev / n.sqrt();
            let z_score = diff_mean / std_err;

            print!("{:40} ", name);
            print!("{:10} ", n);
            print!(
                "{:>10} {:>10} ",
                HumanTime(base_min as f64),
                HumanTime(candidate_min as f64)
            );
            print!(
                "{:>10} {:>10} ",
                HumanTime(base_max as f64),
                HumanTime(candidate_max as f64)
            );
            let min_diff = (candidate_min - base_min) as f64 / base_min as f64 * 100.;
            print!("{:9.1}% ", min_diff);
            print!(
                "{:>10} {:>10} ",
                HumanTime(base_mean),
                HumanTime(candidate_mean)
            );
            print!("{:>10} ", HumanTime(diff_mean));
            print!("{:9.1}% ", diff_mean / base_mean * 100.);
            print!(
                "{:5} {:4.1}% ",
                filtered,
                filtered as f64 / (n as f64) * 100.
            );
            if z_score.abs() >= 2.6 {
                if diff_mean > 0. {
                    print!("CANDIDATE SLOWER");
                } else {
                    print!("CANDIDATE FASTER");
                }
            }
            println!();

            if self.write_data {
                let file_name = format!("{}-{}.csv", baseline_name, candidate_name);
                let mut file = BufWriter::new(File::create(file_name).unwrap());

                // Writing at most 1000 points to csv file. GNUplot can't handle more
                let factor = 1.max(base.len() / 1000);

                for i in 0..base.len() {
                    if i % factor == 0 {
                        writeln!(&mut file, "{},{}", base[i], candidate[i]).unwrap();
                    }
                }
            }
        }

        fn before_start(&mut self) {
            if !self.header_printed {
                self.header_printed = true;
                println!(
                    "{:40} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>11}",
                    "name",
                    "iterations",
                    "B min",
                    "C min",
                    "B max",
                    "C max",
                    "min ∆",
                    "B mean",
                    "C mean",
                    "mean ∆",
                    "mean ∆ (%)",
                    "outliers"
                );
            }
        }
    }

    /// Winsorizing symmetric outliers in a slice
    ///
    /// [Winsorizing][winsorize] is a tchinque of removing outliers in a dataset effectively masking then
    /// with what the most exteme observations left (wo. outliers). This particular algorithm will remove outliers
    /// only if following criteria holds:
    ///
    /// - only 5% of observations are removed from each size
    /// - only outliers greater than 3 IQR from median are removed
    ///
    /// [winsorize]: https://en.wikipedia.org/wiki/Winsorizing
    fn mask_symmetric_outliers(input: &mut [i64]) -> usize {
        let mut filtered = 0;
        let n = input.len();

        let mut sorted = input.to_vec();
        sorted.sort();

        let iqr = sorted[n * 75 / 100] - sorted[n * 25 / 100];

        let mut top = sorted.len() - 1;
        let mut bottom = 0;
        let mut commited_top = top;
        let mut commited_bottom = bottom;

        let median = sorted[sorted.len() / 2];

        while bottom < n * 10 / 100 && top > n * 90 / 100 {
            let bottom_diff = median - sorted[bottom];
            let top_diff = sorted[top] - median;

            let diff = bottom_diff.max(top_diff);
            if diff < 3 * iqr {
                break;
            }

            if top_diff > bottom_diff {
                top -= 1;
            } else {
                bottom += 1;
            }

            let top_removed = n - top - 1;
            let bottom_removed = bottom;
            let abs_diff = top_removed.abs_diff(bottom_removed);

            // TODO Replace this with binomial coefficient/normal distribution approximation calculations
            let deviation = abs_diff as f64 / (bottom_removed + top_removed) as f64;
            if abs_diff < 5 || deviation < 0.3 {
                commited_top = top;
                commited_bottom = bottom;
            }
        }

        for el in input.iter_mut() {
            if *el < sorted[commited_bottom] {
                *el = sorted[commited_bottom];
                filtered += 1;
            } else if *el > sorted[commited_top] {
                *el = sorted[commited_top];
                filtered += 1;
            }
        }

        filtered
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_symmetric_outliers() {
            let mut input = [50i64; 100];
            input[0] = -1000;
            input[1] = -1000;

            mask_symmetric_outliers(&mut input);

            assert_eq!(input[0], 50);
            assert_eq!(input[input.len() - 1], 50);
        }

        #[test]
        fn check_summary_statistics() {
            let stat = Summary::from(vec![1, 1, 2, 4]);
            assert_eq!(stat.min, 1);
            assert_eq!(stat.max, 4);
            assert_eq!(stat.variance, 2.);
        }
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
        let i = f64::from(self.iter.next()? as f64);
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
    input.sort_by(|a, b| a.abs().cmp(&b));
    let variance = RunningVariance::from(input.iter().copied());

    let mut value_and_variance = input
        .iter()
        .copied()
        .zip(variance)
        .skip(input.len() * 30 / 100); // Looking only 30% topmost values

    let mut prev_variance = 0.;
    while let Some((value, var)) = value_and_variance.next() {
        if prev_variance > 0. {
            if var / prev_variance > 2. {
                return Some((-value.abs(), value.abs()));
            }
        }
        prev_variance = var;
    }

    None
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

    struct RngIterator(SmallRng);

    impl Iterator for RngIterator {
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
            101, -102, 103, -104, 105, -106, 107, -108, 109, -110,
        ];

        let (min, max) = outliers_threshold(input).unwrap();
        assert!(min < 1, "Minimum is: {}", min);
        assert!(10 < max && max <= 101, "Maximum is: {}", max);
    }
}
