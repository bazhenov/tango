use core::fmt;
use std::num::NonZeroU64;

use crate::{Benchmark, Reporter};
use clap::Parser;

use self::reporting::{NewConsoleReporter, VerboseReporter};

#[derive(Parser, Debug)]
enum RunMode {
    Pair {
        baseline: String,
        candidates: Vec<String>,

        #[arg(long = "bench", default_value_t = true)]
        bench: bool,
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
    subcommand: RunMode,

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
        RunMode::Pair {
            candidates,
            baseline,
            ..
        } => {
            for candidate in &candidates {
                benchmark.run_pair(&baseline, candidate, reporter);
            }
        }
        RunMode::Calibrate { .. } => {
            benchmark.run_calibration(reporter);
        }
        RunMode::List { .. } => {
            for fn_name in benchmark.list_functions() {
                println!("{}", fn_name);
            }
        }
    }
}

pub mod reporting {
    use crate::cli::HumanTime;
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
            let (base, cand): (Vec<_>, Vec<_>) = measurements
                .iter()
                .copied()
                .map(|(b, c)| (b as i64, c as i64))
                .unzip();

            let base = Summary::from(base);
            let cand = Summary::from(cand);
            let diff = Summary::from(
                measurements
                    .iter()
                    .copied()
                    .map(|(b, c)| c as i64 - b as i64),
            );

            println!("{:12} {:>10} {:>10}", "", baseline, candidate);
            println!("{:12} {:>10} {:>10}", "n", base.n, cand.n);
            println!(
                "{:12} {:>10} {:>10}",
                "min",
                HumanTime(base.min as f64),
                HumanTime(cand.min as f64)
            );
            println!(
                "{:12} {:>10} {:>10}",
                "max",
                HumanTime(base.max as f64),
                HumanTime(cand.max as f64)
            );
            println!(
                "{:12} {:>10} {:>10}",
                "mean",
                HumanTime(base.mean),
                HumanTime(cand.mean)
            );
            println!(
                "{:12} {:>10} {:>10}",
                "std. dev.",
                HumanTime(base.variance.sqrt()),
                HumanTime(cand.variance.sqrt())
            );
            println!();

            println!("{:12} {:>10} {:>10}", "∆ mean", "", HumanTime(diff.mean));
            println!(
                "{:12} {:>10} {:>10}",
                "∆ std. dev.",
                "",
                HumanTime(diff.variance.sqrt())
            );
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
