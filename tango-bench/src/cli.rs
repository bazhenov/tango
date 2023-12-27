//! Contains functionality of a `cargo bench` harness

use self::{
    commands::PairedTest,
    reporting::{ConsoleReporter, VerboseReporter},
};
use crate::{dylib::Spi, Error, MeasurementSettings, Reporter, SamplerType};
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
pub type StdResult<T, E> = std::result::Result<T, E>;

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

        /// seed for the random number generator or omit to use a random seed
        #[arg(long = "seed")]
        seed: Option<u64>,

        #[arg(short = 's', long = "samples")]
        samples: Option<NonZeroUsize>,

        #[arg(long = "sampler")]
        sampler: Option<SamplerType>,

        #[arg(short = 't', long = "time")]
        time: Option<f64>,

        #[arg(long = "fail-threshold")]
        fail_threshold: Option<f64>,

        #[arg(long = "cache-firewall")]
        cache_firewall: Option<bool>,

        #[arg(long = "yield-before-sample")]
        yield_before_sample: Option<bool>,

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

impl FromStr for SamplerType {
    type Err = Error;

    fn from_str(s: &str) -> StdResult<Self, Self::Err> {
        match s {
            "flat" => Ok(SamplerType::Flat),
            "linear" => Ok(SamplerType::Linear),
            "random" => Ok(SamplerType::Random),
            _ => Err(Error::UnknownSamplerType),
        }
    }
}

/// Definition of the flags required to comply with `cargo bench` calling conventions.
#[derive(Parser, Debug, Clone)]
struct CargoBenchFlags {
    #[arg(long = "bench", default_value_t = true)]
    bench: bool,
}

pub fn run(mut settings: MeasurementSettings) -> Result<ExitCode> {
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
            bench_flags: _,
            path,
            verbose,
            filter,
            samples,
            time,
            filter_outliers,
            path_to_dump,
            fail_threshold,
            significant_only,
            seed,
            sampler,
            cache_firewall,
            yield_before_sample,
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

            let spi_self = Spi::for_self().ok_or(Error::SpiSelfWasMoved)??;
            let lib = unsafe { Library::new(&path) }
                .with_context(|| format!("Unable to open library: {}", path.display()))?;
            let spi_lib = Spi::for_library(&lib)?;

            settings.filter_outliers = filter_outliers;
            if let Some(cache_firewall) = cache_firewall {
                settings.cache_firewall = cache_firewall;
            }

            if let Some(yield_before_sample) = yield_before_sample {
                settings.yield_before_sample = yield_before_sample;
            }

            if let Some(sampler) = sampler {
                settings.sampler_type = sampler;
            }

            let filter = filter.as_deref().unwrap_or("");

            let loop_mode = create_loop_mode(samples, time)?;
            let paired_test =
                PairedTest::new(&spi_lib, &spi_self, settings, seed, loop_mode, path_to_dump);

            for func in spi_self.tests() {
                if !filter.is_empty() && !glob_match(filter, &func.name) {
                    continue;
                }

                if spi_lib.lookup(&func.name).is_none() {
                    continue;
                }

                let result = paired_test.run(&func.name)?;

                if result.diff_estimate.significant || !significant_only {
                    reporter.on_complete(&result);
                }

                if result.diff_estimate.significant {
                    if let Some(threshold) = fail_threshold {
                        if result.diff_estimate.pct >= threshold {
                            eprintln!(
                                "[ERROR] Performance regressed {:+.1}% >= {:.1}%  -  test: {}",
                                result.diff_estimate.pct, threshold, func.name
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

fn create_loop_mode(samples: Option<NonZeroUsize>, time: Option<f64>) -> Result<LoopMode> {
    let loop_mode = match (samples, time) {
        (Some(samples), None) => LoopMode::Samples(samples.into()),
        (None, Some(time)) => LoopMode::Time(Duration::from_millis((time * 1000.) as u64)),
        (None, None) => LoopMode::Time(Duration::from_millis(100)),
        (Some(_), Some(_)) => bail!("-t and -s are mutually exclusive"),
    };
    Ok(loop_mode)
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

    use super::*;
    use crate::{
        calculate_run_result, dylib::NamedFunction, CacheFirewall, FlatSampler, LinearSampler,
        RandomSampler, RunResult, Sampler, SamplerType,
    };
    use std::{
        fs::File,
        io::{self, BufWriter, Write as _},
        mem,
        path::Path,
        time::Instant,
    };

    struct TestedFunction<'a> {
        spi: &'a Spi<'a>,
        func: &'a NamedFunction,
        samples: Vec<u64>,
    }

    impl<'a> TestedFunction<'a> {
        fn new(spi: &'a Spi<'a>, func: &'a NamedFunction) -> Self {
            TestedFunction {
                spi,
                func,
                samples: Vec::new(),
            }
        }

        fn run(&mut self, iterations: usize) {
            let sample = self.spi.run(self.func, iterations);
            self.samples.push(sample);
        }

        fn next_haystack(&mut self) {
            self.spi.next_haystack(self.func);
        }

        fn estimate_iterations(&mut self, iterations: u32) -> usize {
            self.spi.estimate_iterations(self.func, iterations)
        }
    }

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
    pub(crate) struct PairedTest<'a> {
        baseline: &'a Spi<'a>,
        candidate: &'a Spi<'a>,
        seed: u64,
        settings: MeasurementSettings,
        loop_mode: LoopMode,
        samples_dump_path: Option<PathBuf>,
        firewall: CacheFirewall,
    }

    impl<'a> PairedTest<'a> {
        pub fn new(
            baseline: &'a Spi<'a>,
            candidate: &'a Spi<'a>,
            settings: MeasurementSettings,
            seed: Option<u64>,
            loop_mode: LoopMode,
            samples_dump_path: Option<PathBuf>,
        ) -> Self {
            let seed = seed.unwrap_or_else(rand::random);
            let firewall = CacheFirewall::new(1024 * 1024);
            Self {
                baseline,
                candidate,
                seed,
                settings,
                loop_mode,
                samples_dump_path,
                firewall,
            }
        }

        pub fn run(&self, test_name: &str) -> Result<RunResult> {
            let a_func = self
                .baseline
                .lookup(test_name)
                .expect("Invalid test name given");
            let b_func = self
                .candidate
                .lookup(test_name)
                .expect("Invalid test name given");

            let mut rng = SmallRng::seed_from_u64(self.seed);

            let seed = rng.next_u64();
            self.baseline.sync(a_func, seed);
            self.candidate.sync(b_func, seed);

            let mut a_func = TestedFunction::new(self.baseline, a_func);
            let mut b_func = TestedFunction::new(self.candidate, b_func);

            // Estimating the number of iterations achievable in 1 ms
            let iterations_per_sample =
                b_func.estimate_iterations(50) / 2 + a_func.estimate_iterations(50) / 2;
            let mut sampler = create_sampler(&self.settings, iterations_per_sample, seed);

            let mut i = 0;
            let mut switch_counter = 0;

            let mut sample_iterations = vec![];

            let start_time = Instant::now();
            while self.loop_mode.should_continue(i, start_time) {
                let iterations = sampler.next_sample_iterations(i);
                i += 1;

                // !!! IMPORTANT !!!
                // Algorithms should be called in different order on each new iteration.
                // This equalize the probability of facing unfortunate circumstances like cache misses or page faults
                // for both functions. Although both algorithms are from distinct shared objects and therefore
                // must be fully self-contained in terms of virtual address space (each shared object has its own
                // generator instances, static variables, memory mappings, etc.) it might be the case that
                // on the level of physical memory both of them rely on the same memory-mapped test data, for example.
                // In that case first function will experience the larger amount of major page faults.
                {
                    mem::swap(&mut a_func, &mut b_func);
                    switch_counter += 1;
                }

                if i % self.settings.samples_per_haystack == 0 {
                    a_func.next_haystack();
                    b_func.next_haystack();
                    if self.settings.cache_firewall {
                        self.firewall.issue_read();
                    }
                }

                if self.settings.yield_before_sample {
                    std::thread::yield_now();
                }

                a_func.run(iterations);
                b_func.run(iterations);
                sample_iterations.push(iterations);
            }

            // If we switched functions odd number of times then we need to swap them back so that
            // the first function is always the baseline.
            if switch_counter % 2 != 0 {
                mem::swap(&mut a_func, &mut b_func);
            }

            let run_result = calculate_run_result(
                test_name,
                &a_func.samples,
                &b_func.samples,
                &sample_iterations,
                self.settings.filter_outliers,
            )
            .ok_or(Error::NoMeasurements)?;

            if let Some(path) = &self.samples_dump_path {
                let file_name = format!("{}.csv", test_name.replace('/', "-"));
                let file_path = path.join(file_name);
                let values = a_func
                    .samples
                    .iter()
                    .copied()
                    .zip(b_func.samples.iter().copied())
                    .zip(sample_iterations.iter().copied())
                    .map(|((a, b), c)| (a, b, c));
                write_raw_measurements(file_path, values)
                    .context("Unable to write raw measurements")?;
            }

            Ok(run_result)
        }
    }

    fn create_sampler(
        settings: &MeasurementSettings,
        estimate: usize,
        seed: u64,
    ) -> Box<dyn Sampler> {
        match settings.sampler_type {
            SamplerType::Flat => Box::new(FlatSampler::new(settings, estimate)),
            SamplerType::Linear => Box::new(LinearSampler::new(settings, estimate)),
            SamplerType::Random => Box::new(RandomSampler::new(settings, estimate, seed)),
        }
    }

    fn write_raw_measurements<A: Display, B: Display, C: Display>(
        path: impl AsRef<Path>,
        values: impl IntoIterator<Item = (A, B, C)>,
    ) -> io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);

        for (a, b, c) in values {
            writeln!(&mut file, "{},{},{}", a, b, c)?;
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

            let significant = results.diff_estimate.significant;

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
                    results.diff_estimate.pct,
                    significant,
                    results.diff.mean < 0.
                ),
                colorize("%", significant, results.diff.mean < 0.),
                if significant { "*" } else { "" },
            );
            println!(
                "    {:12} │ {:>15} {:>15} {:>15}",
                "min",
                HumanTime(base.min),
                HumanTime(candidate.min),
                HumanTime(candidate.min - base.min)
            );
            println!(
                "    {:12} │ {:>15} {:>15} {:>15}",
                "max",
                HumanTime(base.max),
                HumanTime(candidate.max),
                HumanTime(candidate.max - base.max),
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

            let significant = results.diff_estimate.significant;

            let speedup = results.diff_estimate.pct;
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
