//! Contains functionality of a `cargo bench` harness

use self::reporting::{ConsoleReporter, VerboseReporter};
use crate::{dylib::Spi, Error, MeasurementSettings, Reporter};
use anyhow::{bail, Context};
use clap::Parser;
use colorz::mode::{self, Mode};
use core::fmt;
use glob_match::glob_match;
use libloading::Library;
use rand::{rngs::SmallRng, SeedableRng};
use std::{
    env::args,
    fmt::Display,
    num::NonZeroUsize,
    path::PathBuf,
    process::ExitCode,
    str::FromStr,
    time::{Duration, Instant},
};

pub type Result<T> = anyhow::Result<T>;

#[derive(Parser, Debug)]
enum BenchmarkMode {
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
        time: Option<f64>,

        #[arg(long = "fail-threshold")]
        fail_threshold: Option<f64>,

        #[arg(short = 'f', long = "filter")]
        filter: Option<String>,

        // report only statistically significant results
        #[arg(short = 'g', long = "significant-only", default_value_t = false)]
        significant_only: bool,

        /// disable outlier detection
        #[arg(short = 'o', long = "filter-outliers")]
        filter_outliers: bool,

        #[arg(short = 'v', long = "verbose", default_value_t = false)]
        verbose: bool,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Opts {
    #[command(subcommand)]
    subcommand: Option<BenchmarkMode>,

    #[command(flatten)]
    bench_flags: CargoBenchFlags,

    #[arg(long = "color", default_value = "detect")]
    coloring_mode: String,
}

/// Definition of the flags required to comply with `cargo bench` calling conventions.
#[derive(Parser, Debug, Clone)]
struct CargoBenchFlags {
    #[arg(long = "bench", default_value_t = true)]
    bench: bool,
}

pub fn run(settings: MeasurementSettings) -> Result<ExitCode> {
    let opts = Opts::parse();

    match Mode::from_str(&opts.coloring_mode) {
        Ok(coloring_mode) => mode::set_coloring_mode(coloring_mode),
        Err(_) => eprintln!("[WARN] Invalid coloring mode: {}", opts.coloring_mode),
    }

    let subcommand = opts.subcommand.unwrap_or(BenchmarkMode::List {
        bench_flags: opts.bench_flags,
    });

    match subcommand {
        BenchmarkMode::List { bench_flags: _ } => {
            let spi = Spi::for_self().ok_or(Error::SpiSelfWasMoved)??;
            for func in spi.tests() {
                println!("{}", func.name);
            }
            Ok(ExitCode::SUCCESS)
        }
        BenchmarkMode::Compare {
            path,
            verbose,
            filter,
            samples,
            time,
            filter_outliers,
            path_to_dump,
            fail_threshold,
            bench_flags: _,
            significant_only,
        } => {
            let mut reporter: Box<dyn Reporter> = if verbose {
                Box::<VerboseReporter>::default()
            } else {
                Box::<ConsoleReporter>::default()
            };

            let path = path
                .or_else(|| args().next().map(PathBuf::from))
                .expect("No path given");

            #[cfg(target_os = "linux")]
            let path = crate::linux::patch_pie_binary_if_needed(&path)?.unwrap_or(path);

            let lib = unsafe { Library::new(&path) }
                .with_context(|| format!("Unable to open library: {}", path.display()))?;
            let spi_lib = Spi::for_library(&lib)?;
            let spi_self = Spi::for_self().ok_or(Error::SpiSelfWasMoved)??;

            let mut settings = settings;

            let loop_mode = match (samples, time) {
                (Some(samples), None) => LoopMode::Samples(samples.into()),
                (None, Some(time)) => LoopMode::Time(Duration::from_millis((time * 1000.) as u64)),
                (None, None) => LoopMode::Time(Duration::from_millis(100)),
                (Some(_), Some(_)) => bail!("-t and -s are mutually exclusive"),
            };

            settings.filter_outliers = filter_outliers;

            let mut rng = SmallRng::from_entropy();
            let filter = filter.as_deref().unwrap_or("");
            for func in spi_self.tests() {
                if spi_lib.lookup(&func.name).is_none() {
                    continue;
                }
                if !filter.is_empty() && !glob_match(filter, &func.name) {
                    continue;
                }
                let result = commands::paired_compare(
                    &spi_lib,
                    &spi_self,
                    func.name.as_str(),
                    &mut rng,
                    &settings,
                    loop_mode,
                    path_to_dump.as_ref(),
                )?;

                if result.significant || !significant_only {
                    reporter.on_complete(&result);
                }

                if result.significant {
                    if let Some(threshold) = fail_threshold {
                        let diff = result.diff.mean / result.baseline.mean * 100.;
                        if diff >= threshold {
                            eprintln!(
                                "[ERROR] Performance regressed {:+.1}% >= {:.1}%  -  test: {}",
                                diff, threshold, func.name
                            );
                            return Ok(ExitCode::FAILURE);
                        }
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

#[derive(Clone, Copy)]
enum LoopMode {
    Samples(usize),
    Time(Duration),
}

impl LoopMode {
    fn should_continue(&self, iter_no: usize, start_time: Instant) -> bool {
        match self {
            LoopMode::Samples(samples) => iter_no < *samples,
            LoopMode::Time(duration) => {
                // Trying not to stress benchmarking loop with to much of clock calls and check deadline
                // approximately each 8 milliseconds based on the number of iterations already performed
                // (we're assuming each iteration is approximately 1 ms)
                if (iter_no & 0b111) == 0 {
                    Instant::now() < (start_time + *duration)
                } else {
                    true
                }
            }
        }
    }
}

mod commands {
    use rand::RngCore;

    use crate::{calculate_run_result, RunResult};
    use std::{
        fs::File,
        io::{self, BufWriter, Write as _},
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
    ///
    /// Returns a percentage difference in performance of two functions if this change is
    /// statistically significant
    pub(super) fn paired_compare(
        a: &Spi,
        b: &Spi,
        test_name: &str,
        rng: &mut SmallRng,
        settings: &MeasurementSettings,
        loop_mode: LoopMode,
        samples_dump_path: Option<impl AsRef<Path>>,
    ) -> Result<RunResult> {
        let a_func = a.lookup(test_name).expect("Invalid test name given");
        let b_func = b.lookup(test_name).expect("Invalid test name given");

        // Number of iterations estimated based on the performance of A algorithm only. We assuming
        // both algorithms performs approximatley the same. We need to divide estimation by 2 to compensate
        // for the fact that 2 algorithms will be executed concurrently.
        let estimate = a.estimate_iterations(a_func, 1) / 2;
        let iterations_per_ms = estimate.clamp(
            settings.min_iterations_per_sample.max(1),
            settings.max_iterations_per_sample,
        );
        let iterations = iterations_per_ms;

        let mut a_samples = vec![];
        let mut b_samples = vec![];

        let seed = rng.next_u64();
        a.sync(a_func, seed);
        b.sync(b_func, seed);

        let start_time = Instant::now();

        let mut i = 0;
        while loop_mode.should_continue(i, start_time) {
            i += 1;
            if i % settings.samples_per_haystack == 0 {
                a.next_haystack(a_func);
                b.next_haystack(b_func);
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
                let a_time = a.run(a_func, iterations);
                let b_time = b.run(b_func, iterations);

                (a_time, b_time)
            } else {
                let b_time = b.run(b_func, iterations);
                let a_time = a.run(a_func, iterations);

                (a_time, b_time)
            };

            a_samples.push(a_time);
            b_samples.push(b_time);
        }

        if let Some(path) = samples_dump_path {
            let file_name = format!("{}.csv", test_name.replace("/", "-"));
            let file_path = path.as_ref().join(file_name);
            let values = a_samples
                .iter()
                .copied()
                .zip(b_samples.iter().copied())
                .map(|(a, b)| (a / iterations as u64, b / iterations as u64));
            write_raw_measurements(file_path, values)
                .context("Unable to write raw measurements")?;
        }

        Ok(calculate_run_result(
            test_name,
            a_samples,
            b_samples,
            iterations,
            settings.filter_outliers,
        )?)
    }

    fn write_raw_measurements<T: Display>(
        path: impl AsRef<Path>,
        values: impl IntoIterator<Item = (T, T)>,
    ) -> io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);

        for (a, b) in values {
            writeln!(&mut file, "{},{}", a, b)?;
        }
        Ok(())
    }
}

pub mod reporting {
    use crate::cli::{colorize, HumanTime};
    use crate::{Reporter, RunResult};
    use colorz::{mode::Stream, Colorize};

    #[derive(Default)]
    pub(super) struct VerboseReporter;

    impl Reporter for VerboseReporter {
        fn on_complete(&mut self, results: &RunResult) {
            let base = results.baseline;
            let candidate = results.candidate;

            let significant = results.significant;

            println!(
                "{}  (n: {}, outliers: {})",
                results.name.bold().stream(Stream::Stdout),
                results.diff.n,
                results.outliers
            );

            println!(
                "    {:12}   {:>15} {:>15} {:>15}",
                "",
                "baseline".bold().stream(Stream::Stdout),
                "candidate".bold().stream(Stream::Stdout),
                "∆".bold().stream(Stream::Stdout),
            );
            println!(
                "    {:12} ╭────────────────────────────────────────────────",
                ""
            );
            println!(
                "    {:12} │ {:>15} {:>15} {:>15}  {:+4.2}{}{}",
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
                colorize("%", significant, results.diff.mean < 0.),
                if significant { "*" } else { "" },
            );
            println!(
                "    {:12} │ {:>15} {:>15} {:>15}",
                "min",
                HumanTime(base.min as f64),
                HumanTime(candidate.min as f64),
                HumanTime(candidate.min as f64 - base.min as f64)
            );
            println!(
                "    {:12} │ {:>15} {:>15} {:>15}",
                "max",
                HumanTime(base.max as f64),
                HumanTime(candidate.max as f64),
                HumanTime(candidate.max as f64 - base.max as f64),
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
                "{:50} [ {:>8} ... {:>8} ]    {:>+7.2}{}{}",
                colorize(&results.name, significant, candidate_faster),
                HumanTime(base.mean),
                colorize(HumanTime(candidate.mean), significant, candidate_faster),
                colorize(speedup, significant, candidate_faster),
                colorize("%", significant, candidate_faster),
                if significant { "*" } else { "" },
            )
        }
    }
}

fn colorize<T: Display>(value: T, do_paint: bool, is_improved: bool) -> impl Display {
    use colorz::{ansi, mode::Stream::Stdout, Colorize, Style};

    const RED: Style = Style::new().fg(ansi::Red).const_into_runtime_style();
    const GREEN: Style = Style::new().fg(ansi::Green).const_into_runtime_style();
    const DEFAULT: Style = Style::new().const_into_runtime_style();

    if do_paint {
        if is_improved {
            value.into_style_with(GREEN).stream(Stdout)
        } else {
            value.into_style_with(RED).stream(Stdout)
        }
    } else {
        value.into_style_with(DEFAULT).stream(Stdout)
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
        } else if self.0 == 0. {
            f.pad("0 ns")
        } else {
            f.pad(&format!("{:.1} ns", self.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_human_time() {
        assert_eq!(format!("{}", HumanTime(0.1)), "0.1 ns");
        assert_eq!(format!("{:>5}", HumanTime(0.)), " 0 ns");

        assert_eq!(format!("{}", HumanTime(120.)), "120.0 ns");

        assert_eq!(format!("{}", HumanTime(1200.)), "1.2 us");

        assert_eq!(format!("{}", HumanTime(1200000.)), "1.2 ms");

        assert_eq!(format!("{}", HumanTime(1200000000.)), "1.2 s");

        assert_eq!(format!("{}", HumanTime(-1200000.)), "-1.2 ms");
    }
}
