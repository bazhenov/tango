//! Contains functionality of a `cargo bench` harness
use crate::{protocol::WORKER_COMMAND, worker, Benchmark, Error};
use anyhow::{bail, Context};
use clap::{ArgAction, Parser};
use colorz::mode::{self, Mode};
use core::fmt;
use glob_match::glob_match;
use std::{
    env::{self, args, temp_dir},
    fmt::Display,
    fs,
    io::{stderr, Write},
    num::NonZeroUsize,
    ops::Deref,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    str::FromStr,
    time::Duration,
};

pub type Result<T> = anyhow::Result<T>;

#[derive(Parser, Debug)]
enum BenchmarkMode {
    /// List benchmarks
    List {
        #[command(flatten)]
        bench_flags: CargoBenchFlags,
    },
    /// Run paired benchmarking to compare two executables
    Compare(PairedOpts),
    /// Run a single benchmark in a solo (isolated) mode
    Solo(SoloOpts),
}

#[derive(Parser, Debug)]
struct PairedOpts {
    #[command(flatten)]
    bench_flags: CargoBenchFlags,

    /// Path to the executable to test against. Tango will test against itself if no executable given
    path: Option<PathBuf>,

    /// write CSV dumps of all the measurements in a given location
    #[arg(short = 'd', long = "dump")]
    path_to_dump: Option<PathBuf>,

    /// generate gnuplot graphs for each test (requires --dump [path] to be specified)
    #[arg(long = "gnuplot")]
    gnuplot: bool,

    /// seed for the random number generator or omit to use a random seed
    #[arg(long = "seed")]
    seed: Option<u64>,

    /// Number of samples to take for each test
    #[arg(short = 's', long = "samples")]
    samples: Option<NonZeroUsize>,

    /// Duration of each sample in seconds
    #[arg(short = 't', long = "time")]
    time: Option<f64>,

    /// Consider benchmark differences below this percentage as noise (default: 0.5%)
    #[arg(long = "noise-threshold", default_value_t = 0.5)]
    noise_threshold: f64,

    /// Terminate early on first statistically significant performance regression
    #[arg(long = "fail-fast")]
    fail_fast: bool,

    /// Filter tests by name (eg. '*/{sorted,unsorted}/[0-9]*')
    #[arg(short = 'f', long = "filter")]
    filter: Option<String>,

    /// Report only statistically significant results
    #[arg(short = 'g', long = "significant-only", default_value_t = false)]
    significant_only: bool,

    /// Enable outlier detection
    #[arg(short = 'o', long = "filter-outliers")]
    filter_outliers: bool,

    /// Quiet mode
    #[arg(short = 'q')]
    quiet: bool,

    #[arg(short = 'v', long = "verbose", default_value_t = false)]
    verbose: bool,

    /// Disables checking proportion of the time spent in a system/kernel mode
    #[arg(long = "no-system-time-check", default_value_t = true, action = ArgAction::SetFalse)]
    system_time_check: bool,
}

#[derive(Parser, Debug)]
struct SoloOpts {
    #[command(flatten)]
    bench_flags: CargoBenchFlags,

    /// seed for the random number generator or omit to use a random seed
    #[arg(long = "seed")]
    seed: Option<u64>,

    /// Number of samples to take for each test
    #[arg(short = 's', long = "samples")]
    samples: Option<NonZeroUsize>,

    /// Duration of each sample in seconds
    #[arg(short = 't', long = "time")]
    time: Option<f64>,

    /// Filter tests by name (eg. '*/{sorted,unsorted}/[0-9]*')
    #[arg(short = 'f', long = "filter")]
    filter: Option<String>,
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
    #[arg(long = "bench", default_value_t = true, hide = true)]
    bench: bool,
}

pub fn run(benchmarks: Vec<Benchmark>) -> Result<ExitCode> {
    // Check for worker mode before normal CLI parsing
    if env::args().any(|a| a == WORKER_COMMAND) {
        worker::run_worker(benchmarks)
    } else {
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
                for bench in &benchmarks {
                    println!("{}", bench.name());
                }
                Ok(ExitCode::SUCCESS)
            }
            BenchmarkMode::Compare(opts) => paired_test::run_test(opts),
            BenchmarkMode::Solo(opts) => solo_test::run_test(opts, benchmarks),
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

mod solo_test {
    use super::*;
    use crate::Summary;

    pub(super) fn run_test(opts: SoloOpts, mut benchmarks: Vec<Benchmark>) -> Result<ExitCode> {
        let SoloOpts {
            bench_flags: _,
            filter,
            samples,
            time,
            seed,
        } = opts;

        let filter = filter.as_deref().unwrap_or("");
        let loop_mode = create_loop_mode(samples, time)?;
        let seed = seed.unwrap_or_else(rand::random);

        for bench in &mut benchmarks {
            let name = bench.name().to_string();
            if !filter.is_empty() && !glob_match(filter, &name) {
                continue;
            }

            let mut sampler = bench.prepare_state(seed);
            let iters = sampler.estimate_iterations(10);
            let iterations = iters.max(1);

            let mut sample_values = vec![];
            let mut loop_time = Duration::from_secs(0);
            let mut i = 0;

            if let LoopMode::Samples(n) = loop_mode {
                sample_values.reserve(n);
            }

            while loop_mode.should_continue(i, loop_time) {
                let elapsed_ns = sampler.measure(iterations);
                let per_iter = elapsed_ns as f64 / iterations as f64;
                sample_values.push(per_iter);
                loop_time += Duration::from_nanos(elapsed_ns);
                i += 1;
            }

            let result = Summary::from(&sample_values).ok_or(Error::NoMeasurements)?;
            reporting::default_reporter_solo(&name, &result);
        }

        Ok(ExitCode::SUCCESS)
    }
}

mod paired_test {
    use super::*;
    use crate::{
        calculate_run_result,
        child::ChildHandle,
        cli::reporting::BenchmarkProgress,
        commpage::{Commpage, Role},
        platform::RUsage,
    };
    use fs::File;
    use reporting::Reporter;
    use std::{
        io::{self, BufWriter},
        time::Instant,
    };

    pub(super) fn run_test(opts: PairedOpts) -> Result<ExitCode> {
        let PairedOpts {
            bench_flags: _,
            path,
            verbose,
            filter,
            samples,
            time,
            filter_outliers,
            path_to_dump,
            gnuplot,
            noise_threshold,
            fail_fast,
            significant_only,
            seed,
            quiet,
            system_time_check,
        } = opts;
        let mut path = path
            .or_else(|| args().next().map(PathBuf::from))
            .expect("No path given");
        if path.is_relative() {
            if let Ok(pwd) = env::current_dir() {
                path = pwd.join(path)
            }
        };

        if !fs::exists(&path)? {
            let description = format!("Benchmark not found: {}", path.display());
            return Err(io::Error::new(io::ErrorKind::NotFound, description).into());
        }

        let filter = filter.as_deref().unwrap_or("");
        let seed = seed.unwrap_or_else(rand::random);
        let loop_mode = create_loop_mode(samples, time)?;

        let commpage =
            Commpage::create().map_err(|e| anyhow::anyhow!("Failed to create commpage: {e}"))?;

        let candidate_exe = env::current_exe().context("Unable to determine current executable")?;

        let (mut child_c, c_benchmarks) =
            ChildHandle::spawn(&candidate_exe, &commpage, Role::Candidate)
                .context("Failed to spawn candidate")?;
        let (mut child_b, b_benchmarks) = ChildHandle::spawn(&path, &commpage, Role::Baseline)
            .context("Failed to spawn baseline")?;

        let mut exit_code = ExitCode::SUCCESS;

        if let Some(path) = &path_to_dump {
            if !path.exists() {
                fs::create_dir_all(path)?;
            }
        }
        if gnuplot && path_to_dump.is_none() {
            eprintln!("warn: --gnuplot requires -d to be specified. No plots will be generated")
        }

        let mut sample_dumps = vec![];

        let reporter: &dyn Reporter = if verbose {
            &reporting::VerboseReporter
        } else {
            &reporting::DefaultReporter
        };

        for (c_idx, func_name) in c_benchmarks.iter().enumerate() {
            if !filter.is_empty() && !glob_match(filter, func_name) {
                continue;
            }

            let b_idx = match b_benchmarks.iter().position(|n| n == func_name) {
                Some(idx) => idx,
                None => {
                    if !quiet {
                        writeln!(stderr(), "{} skipped...", func_name)?;
                    }
                    continue;
                }
            };

            // Reset commpage and read positions for the new benchmark
            commpage.reset();
            child_c.reset_read_pos();
            child_b.reset_read_pos();

            // Estimate iterations
            const TIME_SLICE_MS: u32 = 10;
            let c_iters = child_c
                .estimate_iterations(c_idx, seed, TIME_SLICE_MS)
                .context("Failed to estimate iterations (candidate)")?;
            let b_iters = child_b
                .estimate_iterations(b_idx, seed, TIME_SLICE_MS)
                .context("Failed to estimate iterations (baseline)")?;
            let iterations = c_iters.max(1).min(b_iters.max(1));

            // Determine num_samples
            let num_samples = match loop_mode {
                LoopMode::Samples(n) => n,
                LoopMode::Time(_) => 0,
            };

            // Start measurement on both children (non-blocking)
            child_c.start_benchmark(c_idx, seed, iterations, num_samples)?;
            child_b.start_benchmark(b_idx, seed, iterations, num_samples)?;

            // Poll samples from the commpage while children are running.
            // The ring buffer only holds 128 samples per lane, so R must drain
            // faster than the children produce to avoid losing data.
            let mut c_samples = Vec::new();
            let mut b_samples = Vec::new();
            let lane_c = commpage.get_lane(Role::Candidate);
            let lane_b = commpage.get_lane(Role::Baseline);

            let time_budget_start = if let LoopMode::Time(_) = loop_mode {
                Some(Instant::now())
            } else {
                None
            };

            loop {
                // Drain whatever is available
                child_c.drain_samples(&commpage, &mut c_samples);
                child_b.drain_samples(&commpage, &mut b_samples);

                let c_done = lane_c.is_done();
                let b_done = lane_b.is_done();

                // In time-budget mode, check if we should signal stop
                if let (Some(start), LoopMode::Time(duration)) = (time_budget_start, loop_mode) {
                    let elapsed = start.elapsed();
                    if elapsed >= duration {
                        commpage.set_stop();
                    } else if elapsed.as_millis() > 50
                        && (duration > Duration::from_millis(500)
                            || elapsed > Duration::from_millis(500))
                    {
                        let phase = BenchmarkProgress::SamplingTime {
                            loop_time: elapsed,
                            total_duration: duration,
                        };
                        reporter.report_progress(func_name, phase);
                    }
                }

                if c_done && b_done {
                    break;
                }

                std::thread::sleep(Duration::from_millis(1));
            }

            // Both children are done — read their RPC responses
            child_c
                .finish_benchmark()
                .context("Candidate benchmark failed")?;
            child_b
                .finish_benchmark()
                .context("Baseline benchmark failed")?;

            // Final drain to pick up any samples written between last poll and DONE
            child_c.drain_samples(&commpage, &mut c_samples);
            child_b.drain_samples(&commpage, &mut b_samples);

            assert!(
                c_samples.len() == b_samples.len() && !c_samples.is_empty(),
                "Invalid number of samples collected: candidate = {}, baseline = {}",
                c_samples.len(),
                b_samples.len()
            );
            let n = c_samples.len();

            let run_result = calculate_run_result(
                &b_samples[..n],
                &c_samples[..n],
                iterations,
                filter_outliers,
                noise_threshold,
            )
            .ok_or(Error::NoMeasurements)?;

            if let Some(path) = &path_to_dump {
                let file_name = format!("{}.csv", func_name.replace('/', "-"));
                let file_path = path.join(file_name);
                write_csv(
                    &file_path,
                    b_samples.iter().zip(c_samples.iter()),
                    iterations,
                )
                .context("Unable to write raw measurements")?;
                sample_dumps.push(file_path);
            }

            if run_result.diff_estimate.significant || !significant_only {
                reporter.benchmark_finished(func_name, &run_result);
            }

            if run_result.diff_estimate.significant && run_result.diff_estimate.pct > 0. {
                exit_code = ExitCode::FAILURE;
                if fail_fast {
                    let _ = child_c.shutdown();
                    let _ = child_b.shutdown();
                    return Ok(ExitCode::FAILURE);
                }
            }
        }

        if gnuplot && !sample_dumps.is_empty() {
            generate_plots(sample_dumps.as_slice())?;
        }

        let _ = child_c.shutdown();
        let _ = child_b.shutdown();

        Ok(exit_code)
    }

    /// Checking if test spent too much time in a system/kernel mode.
    fn detect_system_time_bias(rusage: &RUsage) -> bool {
        let system = rusage.system_time.as_secs_f64();
        let overall = (rusage.user_time + rusage.system_time).as_secs_f64();
        system / overall > 0.05
    }

    fn write_csv<A: Display, B: Display>(
        path: impl AsRef<Path>,
        values: impl IntoIterator<Item = (A, B)>,
        iterations: usize,
    ) -> io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        for (a, b) in values {
            writeln!(&mut file, "{},{},{iterations}", a, b)?;
        }
        Ok(())
    }

    fn generate_plots(sample_dumps: &[PathBuf]) -> Result<()> {
        let gnuplot_file = AutoDelete(temp_dir().join("tango-plot.gnuplot"));
        fs::write(&*gnuplot_file, include_bytes!("plot.gnuplot"))?;
        let gnuplot_file_str = gnuplot_file.to_str().unwrap();

        for input in sample_dumps {
            let csv_input = input.to_str().unwrap();
            let svg_path = input.with_extension("svg");
            let cmd = Command::new("gnuplot")
                .args([
                    "-c",
                    gnuplot_file_str,
                    csv_input,
                    svg_path.to_str().unwrap(),
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .status()
                .context("Failed to execute gnuplot")?;

            if !cmd.success() {
                bail!("gnuplot command failed");
            }
        }
        Ok(())
    }
}

// Automatically removes a file when goes out of scope
struct AutoDelete(PathBuf);

impl Deref for AutoDelete {
    type Target = PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for AutoDelete {
    fn drop(&mut self) {
        if let Err(e) = fs::remove_file(&self.0) {
            eprintln!("Failed to delete file {}: {}", self.0.display(), e);
        }
    }
}

mod reporting {
    use crate::{
        cli::{colorize, HumanTime},
        platform::RUsage,
        RunResult, Summary,
    };
    use colorz::{ansi, mode::Stream, Colorize, Style};
    use std::{
        io::{self, Write},
        time::Duration,
    };

    pub(super) enum BenchmarkProgress {
        #[allow(dead_code)]
        SamplingNo {
            sample_no: usize,
            samples_total: usize,
        },
        SamplingTime {
            loop_time: Duration,
            total_duration: Duration,
        },
    }

    fn clear_progress_line() {
        eprint!("\r\x1b[2K");
        let _ = io::stderr().flush();
    }

    pub(super) trait Reporter {
        fn report_progress(&self, name: &str, progress: BenchmarkProgress) {
            const BAR_WIDTH: usize = 23;
            match progress {
                BenchmarkProgress::SamplingNo {
                    sample_no,
                    samples_total,
                } => {
                    let sample_no = sample_no.min(samples_total);
                    let filled = (sample_no * BAR_WIDTH) / samples_total;
                    let empty = BAR_WIDTH - filled;
                    eprint!(
                        "\r\x1b[2K{:50} [{}{}] {}/{}",
                        name,
                        "#".repeat(filled),
                        ".".repeat(empty),
                        sample_no,
                        samples_total,
                    );
                }
                BenchmarkProgress::SamplingTime {
                    loop_time,
                    total_duration,
                } => {
                    let filled = loop_time.as_millis() as usize * BAR_WIDTH
                        / total_duration.as_millis() as usize;
                    let empty = BAR_WIDTH - filled;
                    eprint!(
                        "\r\x1b[2K{:50} [{}{}] {:.1}s",
                        name,
                        "#".repeat(filled),
                        ".".repeat(empty),
                        loop_time.as_secs_f32(),
                    );
                }
            }
            let _ = io::stderr().flush();
        }

        fn benchmark_finished(&self, name: &str, results: &RunResult);
    }

    pub(super) struct VerboseReporter;

    impl Reporter for VerboseReporter {
        fn benchmark_finished(&self, name: &str, results: &RunResult) {
            clear_progress_line();

            let base = results.baseline;
            let candidate = results.candidate;

            let significant = results.diff_estimate.significant;

            println!(
                "{}  (n: {}, outliers: {})",
                name.bold().stream(Stream::Stdout),
                results.diff.n,
                results.outliers
            );

            println!(
                "    {:12}   {:>15} {:>15} {:>15}",
                "",
                "baseline".bold().stream(Stream::Stdout),
                "candidate".bold().stream(Stream::Stdout),
                "\u{2206}".bold().stream(Stream::Stdout),
            );
            println!("    {:12} \u{256d}{}", "", "\u{2500}".repeat(48));
            println!(
                "    {:12} \u{2502} {:>15} {:>15} {:>15}  {:+4.2}{}{}",
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
                "    {:12} \u{2502} {:>15} {:>15} {:>15}",
                "min",
                HumanTime(base.min),
                HumanTime(candidate.min),
                HumanTime(candidate.min - base.min)
            );
            println!(
                "    {:12} \u{2502} {:>15} {:>15} {:>15}",
                "max",
                HumanTime(base.max),
                HumanTime(candidate.max),
                HumanTime(candidate.max - base.max),
            );
            println!(
                "    {:12} \u{2502} {:>15} {:>15} {:>15}",
                "std. dev.",
                HumanTime(base.variance.sqrt()),
                HumanTime(candidate.variance.sqrt()),
                HumanTime(results.diff.variance.sqrt()),
            );
            println!();
        }
    }

    pub(super) struct DefaultReporter;

    impl Reporter for DefaultReporter {
        fn benchmark_finished(&self, name: &str, results: &RunResult) {
            clear_progress_line();

            let base = results.baseline;
            let candidate = results.candidate;
            let diff = results.diff;

            let significant = results.diff_estimate.significant;

            let speedup = results.diff_estimate.pct;
            let candidate_faster = diff.mean < 0.;
            println!(
                "{:50} [ {:>8} ... {:>8} ]    {:>+7.2}{}{}",
                colorize(name, significant, candidate_faster),
                HumanTime(base.mean),
                colorize(HumanTime(candidate.mean), significant, candidate_faster),
                colorize(speedup, significant, candidate_faster),
                colorize("%", significant, candidate_faster),
                if significant { "*" } else { "" },
            )
        }
    }

    pub(super) fn default_reporter_solo(name: &str, results: &Summary<f64>) {
        println!(
            "{:50}  [ {:>8} ... {:>8} ... {:>8} ]  stddev: {:>8}",
            name,
            HumanTime(results.min),
            HumanTime(results.mean),
            HumanTime(results.max),
            HumanTime(results.variance.sqrt()),
        )
    }

    pub(super) fn report_system_time_bias(name: &str, rusage: &RUsage) {
        const RED: Style = Style::new().fg(ansi::Red).const_into_runtime_style();

        eprintln!(
            "{}: {} benchmark spent too much time in system mode (sys: {:?}, usr: {:?}). Results may be inaccurate",
            "WARN".into_style_with(RED).stream(Stream::Stderr),
            name,
            rusage.system_time,
            rusage.user_time
        );
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
