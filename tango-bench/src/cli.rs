//! Contains functionality of a `cargo bench` harness

use self::{
    commands::run_paired_test,
    reporting::{ConsoleReporter, VerboseReporter},
};
use crate::{
    dylib::{Spi, SpiModeKind},
    Error, MeasurementSettings, Reporter, SampleLengthKind,
};
use anyhow::{bail, Context};
use clap::Parser;
use colorz::mode::{self, Mode};
use core::fmt;
use glob_match::glob_match;
use std::{
    env::args,
    fmt::Display,
    io::{stderr, Write},
    num::NonZeroUsize,
    path::PathBuf,
    process::ExitCode,
    str::FromStr,
    time::Duration,
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

        /// Number of samples to take for each test
        #[arg(short = 's', long = "samples")]
        samples: Option<NonZeroUsize>,

        /// The strategy to decide the number of iterations to run for each sample (values: flat, linear, random)
        #[arg(long = "sampler")]
        sampler: Option<SampleLengthKind>,

        /// Duration of each sample in seconds
        #[arg(short = 't', long = "time")]
        time: Option<f64>,

        /// Fail if the difference between the two measurements is greater than the given threshold in percent
        #[arg(long = "fail-threshold")]
        fail_threshold: Option<f64>,

        /// Should we terminate early if --fail-threshold is exceed
        #[arg(long = "fail-fast")]
        fail_fast: bool,

        /// Perform a read of a dummy data between samsples to minimize the effect of cache on the performance
        /// (size in Kbytes)
        #[arg(long = "cache-firewall")]
        cache_firewall: Option<usize>,

        /// Perform a randomized offset to the stack frame for each sample.
        /// (size in bytes)
        #[arg(long = "randomize-stack")]
        randomize_stack: Option<usize>,

        /// Delegate control back to the OS before each sample
        #[arg(long = "yield-before-sample")]
        yield_before_sample: Option<bool>,

        /// Filter tests by name (eg. '*/{sorted,unsorted}/[0-9]*')
        #[arg(short = 'f', long = "filter")]
        filter: Option<String>,

        /// Report only statistically significant results
        #[arg(short = 'g', long = "significant-only", default_value_t = false)]
        significant_only: bool,

        /// Enable outlier detection
        #[arg(short = 'o', long = "filter-outliers")]
        filter_outliers: bool,

        /// Perform warmup iterations before taking measurements (1/10 of sample iterations)
        #[arg(long = "warmup")]
        warmup_enabled: Option<bool>,

        #[arg(short = 'p', long = "parallel")]
        parallel: bool,

        /// Quiet mode
        #[arg(short = 'q')]
        quiet: bool,

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

impl FromStr for SampleLengthKind {
    type Err = Error;

    fn from_str(s: &str) -> StdResult<Self, Self::Err> {
        match s {
            "flat" => Ok(SampleLengthKind::Flat),
            "linear" => Ok(SampleLengthKind::Linear),
            "random" => Ok(SampleLengthKind::Random),
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
            let spi = Spi::for_self(SpiModeKind::Synchronous).ok_or(Error::SpiSelfWasMoved)?;
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
            fail_fast,
            significant_only,
            seed,
            sampler,
            cache_firewall,
            yield_before_sample,
            warmup_enabled,
            parallel,
            quiet,
            randomize_stack,
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

            let mode = if parallel {
                SpiModeKind::Asynchronous
            } else {
                SpiModeKind::Synchronous
            };

            let mut spi_self = Spi::for_self(mode).ok_or(Error::SpiSelfWasMoved)?;
            let mut spi_lib = Spi::for_library(path, mode);

            settings.filter_outliers = filter_outliers;
            settings.cache_firewall = cache_firewall;
            settings.randomize_stack = randomize_stack;

            if let Some(warmup_enabled) = warmup_enabled {
                settings.warmup_enabled = warmup_enabled;
            }
            if let Some(yield_before_sample) = yield_before_sample {
                settings.yield_before_sample = yield_before_sample;
            }
            if let Some(sampler) = sampler {
                settings.sampler_type = sampler;
            }

            let filter = filter.as_deref().unwrap_or("");
            let loop_mode = create_loop_mode(samples, time)?;

            let mut exit_code = ExitCode::SUCCESS;

            let test_names = spi_self
                .tests()
                .iter()
                .map(|t| &t.name)
                .cloned()
                .collect::<Vec<_>>();
            for func_name in test_names {
                if !filter.is_empty() && !glob_match(filter, &func_name) {
                    continue;
                }

                if spi_lib.lookup(&func_name).is_none() {
                    if !quiet {
                        writeln!(stderr(), "{} skipped...", &func_name)?;
                    }
                    continue;
                }

                let result = run_paired_test(
                    &mut spi_lib,
                    &mut spi_self,
                    &func_name,
                    settings,
                    seed,
                    loop_mode,
                    path_to_dump.as_ref(),
                )?;

                if result.diff_estimate.significant || !significant_only {
                    reporter.on_complete(&result);
                }

                if result.diff_estimate.significant {
                    if let Some(threshold) = fail_threshold {
                        if result.diff_estimate.pct >= threshold {
                            eprintln!(
                                "[ERROR] Performance regressed {:+.1}% >= {:.1}%  -  test: {}",
                                result.diff_estimate.pct, threshold, func_name
                            );
                            exit_code = ExitCode::FAILURE;
                            if fail_fast {
                                return Ok(ExitCode::FAILURE);
                            }
                        }
                    }
                }
            }
            Ok(exit_code)
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
    fn should_continue(&self, iter_no: usize, loop_time: Duration) -> bool {
        match self {
            LoopMode::Samples(samples) => iter_no < *samples,
            LoopMode::Time(duration) => loop_time < *duration,
        }
    }
}

mod commands {

    use super::*;
    use crate::{
        calculate_run_result,
        dylib::{FunctionIdx, Spi},
        CacheFirewall, FlatSampleLength, LinearSampleLength, RandomSampleLength, RunResult,
        SampleLength, SampleLengthKind,
    };
    use alloca::with_alloca;
    use rand::{distributions, rngs::SmallRng, Rng, SeedableRng};
    use std::{
        fs::{self, File},
        io::{self, BufWriter},
        mem,
        path::Path,
    };

    struct TestedFunction<'a> {
        spi: &'a mut Spi,
        samples: Vec<u64>,
    }

    impl<'a> TestedFunction<'a> {
        fn new(spi: &'a mut Spi, func: FunctionIdx) -> Self {
            spi.select(func);
            TestedFunction {
                spi,
                samples: Vec::new(),
            }
        }

        fn measure(&mut self, iterations: usize) {
            self.spi.measure(iterations);
        }

        fn read_sample(&mut self) -> u64 {
            let sample = self.spi.read_sample();
            self.samples.push(sample);
            sample
        }

        fn run(&mut self, iterations: usize) -> u64 {
            self.spi.run(iterations)
        }

        fn prepare_state(&mut self, seed: u64) {
            self.spi.prepare_state(seed);
        }

        fn estimate_iterations(&mut self, time_ms: u32) -> usize {
            self.spi.estimate_iterations(time_ms)
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
    pub fn run_paired_test(
        baseline: &mut Spi,
        candidate: &mut Spi,
        test_name: &str,
        settings: MeasurementSettings,
        seed: Option<u64>,
        loop_mode: LoopMode,
        samples_dump_path: Option<&PathBuf>,
    ) -> Result<RunResult> {
        const TIME_SLICE_MS: u32 = 10;

        let firewall = settings
            .cache_firewall
            .map(|s| s * 1024)
            .map(CacheFirewall::new);
        let baseline_func = baseline.lookup(test_name).ok_or(Error::InvalidTestName)?;
        let candidate_func = candidate.lookup(test_name).ok_or(Error::InvalidTestName)?;

        let mut baseline = TestedFunction::new(baseline, baseline_func.idx);
        let mut candidate = TestedFunction::new(candidate, candidate_func.idx);

        let mut a_func = &mut baseline;
        let mut b_func = &mut candidate;

        let seed = seed.unwrap_or_else(rand::random);

        a_func.prepare_state(seed);
        let a_estimate = (a_func.estimate_iterations(TIME_SLICE_MS) / 2).max(1);

        b_func.prepare_state(seed);
        let b_estimate = (b_func.estimate_iterations(TIME_SLICE_MS) / 2).max(1);

        let mut iterations_per_sample = a_estimate.min(b_estimate);
        let mut sampler = create_sampler(&settings, seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let stack_offset_distr = settings
            .randomize_stack
            .map(|offset| distributions::Uniform::new(0, offset));

        let mut i = 0;
        let mut switch_counter = 0;

        let mut sample_iterations = vec![];

        if let LoopMode::Samples(samples) = loop_mode {
            sample_iterations.reserve(samples);
            a_func.samples.reserve(samples);
            b_func.samples.reserve(samples);
        }

        let mut loop_time = Duration::from_secs(0);
        let mut loop_iterations = 0;
        while loop_mode.should_continue(i, loop_time) {
            if loop_time > Duration::from_millis(100) {
                // correcting time slice estimates
                iterations_per_sample =
                    loop_iterations * TIME_SLICE_MS as usize / loop_time.as_millis() as usize;
            }
            let iterations = sampler.next_sample_iterations(i, iterations_per_sample);
            loop_iterations += iterations;
            let warmup_iterations = settings.warmup_enabled.then(|| (iterations / 10).max(1));

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

            if settings.yield_before_sample {
                std::thread::yield_now();
            }

            let prepare_state_seed = (i % settings.samples_per_haystack == 0).then_some(seed);
            let mut sample_time = 0;

            prepare_func(
                prepare_state_seed,
                a_func,
                warmup_iterations,
                firewall.as_ref(),
            );
            prepare_func(
                prepare_state_seed,
                b_func,
                warmup_iterations,
                firewall.as_ref(),
            );

            // Allocate a custom stack frame during runtime, to try to offset alignment of the stack.
            if let Some(distr) = stack_offset_distr {
                with_alloca(rng.sample(distr), |_| {
                    a_func.measure(iterations);
                    b_func.measure(iterations);
                });
            } else {
                a_func.measure(iterations);
                b_func.measure(iterations);
            }

            let a_sample_time = a_func.read_sample();
            let b_sample_time = b_func.read_sample();
            sample_time += a_sample_time.max(b_sample_time);

            loop_time += Duration::from_nanos(sample_time);
            sample_iterations.push(iterations);
            i += 1;
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
            settings.filter_outliers,
        )
        .ok_or(Error::NoMeasurements)?;

        if let Some(path) = samples_dump_path {
            if !path.exists() {
                fs::create_dir_all(path)?;
            }
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

    fn prepare_func(
        prepare_state_seed: Option<u64>,
        f: &mut TestedFunction,
        warmup_iterations: Option<usize>,
        firewall: Option<&CacheFirewall>,
    ) {
        if let Some(seed) = prepare_state_seed {
            f.prepare_state(seed);
            if let Some(firewall) = firewall {
                firewall.issue_read();
            }
        }
        if let Some(warmup_iterations) = warmup_iterations {
            f.run(warmup_iterations);
        }
    }

    fn create_sampler(settings: &MeasurementSettings, seed: u64) -> Box<dyn SampleLength> {
        match settings.sampler_type {
            SampleLengthKind::Flat => Box::new(FlatSampleLength::new(settings)),
            SampleLengthKind::Linear => Box::new(LinearSampleLength::new(settings)),
            SampleLengthKind::Random => Box::new(RandomSampleLength::new(settings, seed)),
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

    // Sane checking some simple patterns
    #[test]
    fn check_glob() {
        let patterns = vec!["a/*/*", "a/**", "*/32/*", "**/b", "a/{32,64}/*"];
        let input = "a/32/b";
        for pattern in patterns {
            assert!(
                glob_match(pattern, input),
                "failed to match {} against {}",
                pattern,
                input
            );
        }
    }
}
