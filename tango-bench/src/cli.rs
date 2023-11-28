use crate::{dylib::Spi, platform, MeasurementSettings, Reporter};
use clap::Parser;
use core::fmt;
use libloading::Library;
use std::{
    collections::HashSet,
    fmt::Display,
    hash::Hash,
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
    time::Duration,
};

use self::reporting::{ConsoleReporter, VerboseReporter};

#[derive(Parser, Debug)]
enum BenchmarkMode {
    Calibrate {
        #[command(flatten)]
        bench_flags: CargoBenchFlags,
    },
    List {
        #[command(flatten)]
        bench_flags: CargoBenchFlags,
    },
    Compare {
        #[command(flatten)]
        bench_flags: CargoBenchFlags,

        /// Path to the executable to test agains. Tango will test agains itself if no executable given
        path: Option<PathBuf>,

        /// write CSV dumps of all the measurements in a given location
        #[arg(short = 'd', long = "dump")]
        path_to_dump: Option<PathBuf>,

        #[arg(short = 's', long = "samples")]
        samples: Option<NonZeroUsize>,

        #[arg(short = 't', long = "time")]
        time: Option<NonZeroU64>,

        #[arg(short = 'f', long = "filter")]
        filter: Option<String>,

        /// disable outlier detection
        #[arg(short = 'o', long = "no-outliers")]
        skip_outlier_detection: bool,

        #[arg(short = 'v', long = "verbose", default_value_t = false)]
        verbose: bool,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Opts {
    #[command(subcommand)]
    subcommand: BenchmarkMode,

    #[command(flatten)]
    bench_flags: CargoBenchFlags,
}

/// Definition of the flags required to comply with `cargo bench` calling conventions.
#[derive(Parser, Debug, Clone)]
struct CargoBenchFlags {
    #[arg(long = "bench", default_value_t = true)]
    bench: bool,
}

pub fn run(settings: MeasurementSettings) {
    let opts = Opts::parse();

    match opts.subcommand {
        BenchmarkMode::Calibrate { bench_flags: _ } => {
            todo!();
            // benchmark.run_calibration();
        }
        BenchmarkMode::List { bench_flags: _ } => {
            let spi = Spi::for_self().unwrap();
            let test_names = spi.tests().keys();
            for name in test_names {
                println!("{}", name);
            }
        }
        BenchmarkMode::Compare {
            path,
            verbose,
            filter,
            samples,
            time,
            skip_outlier_detection,
            path_to_dump,
            bench_flags: _,
        } => {
            let mut reporter: Box<dyn Reporter> = if verbose {
                Box::<VerboseReporter>::default()
            } else {
                Box::<ConsoleReporter>::default()
            };

            let self_path = PathBuf::from(std::env::args().next().unwrap());
            let path = path.unwrap_or(self_path);

            let path = if let Some(replacement) = platform::patch_pie_binary_if_needed(&path) {
                replacement
            } else {
                path
            };
            let lib = unsafe { Library::new(path) }.expect("Unable to load library");
            let spi_lib = Spi::for_library(&lib);
            let spi_self = Spi::for_self().expect("SelfSpi already called once");

            let mut test_names = intersect_values(spi_lib.tests().keys(), spi_self.tests().keys());
            test_names.sort();

            let mut opts = settings;
            if let Some(samples) = samples {
                opts.max_samples = samples.into()
            }
            if let Some(millis) = time {
                opts.max_duration = Duration::from_millis(millis.into());
            }
            if skip_outlier_detection {
                opts.outlier_detection_enabled = false;
            }

            let filter = filter.as_deref().unwrap_or("");
            for name in test_names {
                if !name.contains(filter) {
                    continue;
                }
                commands::pairwise_compare(
                    &spi_self,
                    &spi_lib,
                    name.as_str(),
                    &opts,
                    reporter.as_mut(),
                    path_to_dump.as_ref(),
                );
            }
        }
    }
}

/// Returning the intersection of the values of two iterators buffering them in an intermediate [`HashSet`].
fn intersect_values<'a, K: Hash + Eq>(
    a: impl Iterator<Item = &'a K>,
    b: impl Iterator<Item = &'a K>,
) -> Vec<&'a K> {
    let a_values = a.collect::<HashSet<_>>();
    let b_values = b.collect::<HashSet<_>>();
    a_values
        .intersection(&b_values)
        .copied()
        .collect::<Vec<_>>()
}

mod commands {
    use crate::{calculate_run_result, Summary};
    use std::{
        fs::File,
        io::{BufWriter, Write as _},
        path::Path,
        time::Instant,
    };

    use super::*;

    /// Measure the difference in performance of two functions
    ///
    /// Provides a way to save a raw dump of measurements into directory
    ///
    /// The format is as follows
    /// ```txt
    /// b_1,c_1
    /// b_2,c_2
    /// ...
    /// b_n,c_n
    /// ```
    /// where `b_1..b_n` are baseline absolute time (in nanoseconds) measurements
    /// and `c_1..c_n` are candidate time measurements
    pub(super) fn pairwise_compare(
        a: &Spi,
        b: &Spi,
        test_name: &str,
        settings: &MeasurementSettings,
        reporter: &mut dyn Reporter,
        samples_dump_path: Option<impl AsRef<Path>>,
    ) {
        let a_idx = *a.tests().get(test_name).unwrap();
        let b_idx = *b.tests().get(test_name).unwrap();

        // Number of iterations estimated based on the performance of A algorithm only. We assuming
        // both algorithms performs approximatley the same. We need to divide estimation by 2 to compensate
        // for the fact that 2 algorithms will be executed concurrently.
        let estimate = a.estimate_iterations(a_idx, 1) / 2;
        let iterations_per_ms = estimate.clamp(
            settings.min_iterations_per_sample.max(1),
            settings.max_iterations_per_sample,
        );
        let iterations = iterations_per_ms;

        let mut a_samples = vec![];
        let mut b_samples = vec![];
        let mut diff = vec![];

        let deadline = Instant::now() + settings.max_duration;

        for i in 0..settings.max_samples {
            // Trying not to stress benchmarking loop with to much of clock calls and check deadline
            // approximately each millisecond based on the number of iterations already performed
            if i % iterations_per_ms == 0 && Instant::now() >= deadline {
                break;
            }

            if i % settings.samples_per_haystack == 0 {
                a.next_haystack();
                b.next_haystack();
            }

            // !!! IMPORTANT !!!
            // Algorithms should be called in different order in those two branches.
            // This equalize the probability of facing unfortunate circumstances like cache misses or page faults
            // for both functions. Although both algorithms are from distinct shared objects and therefore
            // must be fully selfcontained in terms of virtual address space (each shared object has its own
            // generator instances, static variables, memory mappings, etc.) it might be the case that
            // on the level of physical memory both of them rely on the same memory-mapped test data, for example.
            // In that case first function will experience the larger amount of major page faults.
            let (a_time, b_time) = if i % 2 == 0 {
                let a_time = a.run(a_idx, iterations);
                let b_time = b.run(b_idx, iterations);

                (a_time, b_time)
            } else {
                let b_time = b.run(b_idx, iterations);
                let a_time = a.run(a_idx, iterations);

                (a_time, b_time)
            };

            a_samples.push(a_time as i64 / iterations as i64);
            b_samples.push(b_time as i64 / iterations as i64);
            diff.push((b_time - a_time) as i64 / iterations as i64);
        }

        if let Some(path) = samples_dump_path {
            let file_name = format!("{}.csv", test_name);
            let file_path = path.as_ref().join(file_name);
            write_raw_measurements(file_path, &a_samples, &b_samples);
        }

        let a_summary = Summary::from(&a_samples).unwrap();
        let b_summary = Summary::from(&b_samples).unwrap();

        let result = calculate_run_result(
            (test_name, a_summary),
            (test_name, b_summary),
            diff,
            settings.outlier_detection_enabled,
        );

        reporter.on_complete(&result);
    }

    fn write_raw_measurements(path: impl AsRef<Path>, base: &[i64], candidate: &[i64]) {
        let mut file = BufWriter::new(File::create(path).unwrap());

        for (b, c) in base.iter().zip(candidate) {
            writeln!(&mut file, "{},{}", b, c).unwrap();
        }
    }
}

pub mod reporting {

    use crate::cli::{colorize, Color, Colored, HumanTime};
    use crate::{Reporter, RunResult};

    #[derive(Default)]
    pub(super) struct VerboseReporter;

    impl Reporter for VerboseReporter {
        fn on_complete(&mut self, results: &RunResult) {
            let base = results.baseline;
            let candidate = results.candidate;

            let significant = results.significant;

            println!(
                "{}  (n: {}, outliers: {})",
                Colored(&results.name, Color::Bold),
                results.diff.n,
                results.outliers
            );

            println!(
                "    {:12}   {:>15} {:>15} {:>15}",
                "",
                Colored("baseline", Color::Bold),
                Colored("candidate", Color::Bold),
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
                "    {:12} │ {:>15} {:>15} {:>15}  {:+4.2}{}",
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
    pub(super) struct ConsoleReporter;

    impl Reporter for ConsoleReporter {
        fn on_complete(&mut self, results: &RunResult) {
            let base = results.baseline;
            let candidate = results.candidate;
            let diff = results.diff;

            let significant = results.significant;

            let speedup = diff.mean / base.mean * 100.;
            let candidate_faster = diff.mean < 0.;
            println!(
                "{:50} [ {:>8} ... {:>8} ]    {:>+7.2}{}",
                colorize(&results.name, significant, candidate_faster),
                HumanTime(base.mean),
                colorize(HumanTime(candidate.mean), significant, candidate_faster),
                colorize(speedup, significant, candidate_faster),
                colorize("%", significant, candidate_faster)
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
