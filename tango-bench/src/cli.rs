//! Contains functionality of a `cargo bench` harness
use crate::{available_aux_metrics, commpage::Role, worker, Benchmark, Error};
use anyhow::{bail, Context};
use clap::{ArgAction, Parser};
use core::fmt;
use glob_match::glob_match;
use std::{
    env::{self, temp_dir},
    fmt::Display,
    fs,
    io::Write,
    num::NonZeroUsize,
    ops::Deref,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    time::Duration,
};

pub type Result<T> = anyhow::Result<T>;

#[derive(Parser, Debug)]
enum Subcommand {
    /// List benchmarks
    List {
        #[command(flatten)]
        bench_flags: CargoBenchFlags,
    },
    /// List auxiliary metrics
    AuxMetrics {
        #[command(flatten)]
        bench_flags: CargoBenchFlags,
    },
    /// Run paired benchmarking to compare two executables
    Compare(PairedOpts),
    /// Run a single benchmark in a solo (isolated) mode
    Solo(SoloOpts),
    /// Internal worker mode (used by the runner to spawn child processes)
    #[command(name = "__worker", hide = true)]
    Worker(WorkerOpts),
}

#[derive(Parser, Debug)]
struct WorkerOpts {
    /// Shared memory name for the commpage
    #[arg(long = "shmem")]
    shmem: String,

    /// Role of this worker (candidate or baseline)
    #[arg(long = "role")]
    role: String,
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

    /// Consider benchmark differences below this percentage as noise (default: 1%)
    #[arg(long = "noise-threshold", default_value_t = 1.0)]
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

    /// Disables reporting a progress of a running benchmark
    #[arg(long = "progress", default_value_t = false, action = ArgAction::SetTrue)]
    progress_report: bool,
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
    subcommand: Option<Subcommand>,

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
    let opts = Opts::parse();

    let subcommand = opts.subcommand.unwrap_or(Subcommand::List {
        bench_flags: opts.bench_flags,
    });

    // Handle worker mode early, before setting up coloring etc.
    if let Subcommand::Worker(ref worker_opts) = subcommand {
        let role: Role = serde_json::from_value(serde_json::Value::String(
            worker_opts.role.clone(),
        ))
        .map_err(|_| {
            anyhow::anyhow!(
                "Invalid role: '{}' (expected 'candidate' or 'baseline')",
                worker_opts.role
            )
        })?;
        worker::run_worker(&worker_opts.shmem, role, benchmarks)
    } else {
        match opts.coloring_mode.as_str() {
            "always" => colored::control::set_override(true),
            "never" => colored::control::set_override(false),
            "detect" | "auto" => colored::control::unset_override(),
            _ => eprintln!("[WARN] Invalid coloring mode: {}", opts.coloring_mode),
        }

        match subcommand {
            Subcommand::Worker(_) => unreachable!(),
            Subcommand::List { bench_flags: _ } => {
                for bench in &benchmarks {
                    println!("{}", bench.name());
                }
                Ok(ExitCode::SUCCESS)
            }
            Subcommand::AuxMetrics { bench_flags: _ } => {
                for m in available_aux_metrics() {
                    println!("{}", m.id);
                }
                Ok(ExitCode::SUCCESS)
            }
            Subcommand::Compare(opts) => paired_test::run_test(opts),
            Subcommand::Solo(opts) => solo_test::run_test(opts, benchmarks),
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
        aux::CPU_SYSTEM_TIME_ID,
        calculate_run_result,
        child::ChildHandle,
        cli::reporting::{report_system_time_bias, BenchmarkProgress},
        commpage::{Commpage, Role},
        protocol::RunBenchmarkResult,
    };
    use fs::File;
    use reporting::Reporter;
    use std::{
        collections::HashSet,
        io::{self, BufWriter},
        thread::sleep,
        time::Instant,
    };

    const TIME_SLICE: Duration = Duration::from_millis(10);

    /// How frequent we report progress to the user
    const PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(50);

    /// If benchmark duration is less than this duration do not report the progress to the user.
    /// Two reasons for this:
    ///  1. when benchmark is fast additional report leads to flickering and worsen the experience
    ///  2. additional reporting may lead to unneeded context switches
    const PROGRESS_REPORTING_THRESHOLD: Duration = Duration::from_secs(1);

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
            progress_report,
        } = opts;

        let candidate_exe = env::current_exe().context("Unable to determine current executable")?;
        // If no baseline executable was given, comparing against itself
        let mut path = path.unwrap_or(candidate_exe.clone());
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

        let mut commpage = Commpage::create().context("Failed to create commpage")?;

        let mut child_c = ChildHandle::spawn(&candidate_exe, &commpage, Role::Candidate)
            .context("Failed to spawn candidate")?;
        let mut child_b = ChildHandle::spawn(&path, &commpage, Role::Baseline)
            .context("Failed to spawn baseline")?;

        let c_list = child_c
            .list_benchmarks()
            .context("Failed to list benchmarks (candidate)")?;
        let b_list = child_b
            .list_benchmarks()
            .context("Failed to list benchmarks (baseline)")?;
        let c_benchmarks = c_list.benchmarks;
        let b_benchmarks = b_list.benchmarks;

        if let Some(path) = &path_to_dump {
            if !path.exists() {
                fs::create_dir_all(path)?;
            }
        }
        if gnuplot && path_to_dump.is_none() && !quiet {
            eprintln!("warn: --gnuplot requires -d to be specified. No plots will be generated")
        }

        let mut sample_dumps = vec![];

        let reporter: &dyn Reporter = if verbose {
            &reporting::VerboseReporter
        } else {
            &reporting::DefaultReporter
        };

        // Determine num_samples
        let num_samples = match loop_mode {
            LoopMode::Samples(n) => n,
            LoopMode::Time(_) => 0,
        };

        let mut exit_code = ExitCode::SUCCESS;

        // Calculate common aux-metric names if verbose mode activated
        let aux_metrics = if verbose {
            let b_metrics = HashSet::<String>::from_iter(b_list.aux_metrics);
            let c_metrics = HashSet::from_iter(c_list.aux_metrics);
            b_metrics
                .intersection(&c_metrics)
                .cloned()
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        let selected_benchmarks = c_benchmarks
            .iter()
            .enumerate()
            // Filtering benchmarks by -f
            .filter(|(_, f)| filter.is_empty() || glob_match(filter, f))
            // finding position of the benchmark in a baseline
            .filter_map(|(c_idx, f)| {
                b_benchmarks
                    .iter()
                    .position(|b_func| b_func == f)
                    .map(|b_idx| (c_idx, b_idx, f))
            });
        for (c_idx, b_idx, func_name) in selected_benchmarks {
            // Estimate iterations
            let c_iters = child_c
                .estimate_iterations(c_idx, seed, TIME_SLICE)
                .context("Failed to estimate iterations (candidate)")?;
            let b_iters = child_b
                .estimate_iterations(b_idx, seed, TIME_SLICE)
                .context("Failed to estimate iterations (baseline)")?;
            let iterations = c_iters.max(1).min(b_iters.max(1));

            commpage.reset();

            // Start measurement on both children (non-blocking)
            child_c.start_benchmark(c_idx, seed, iterations, num_samples, aux_metrics.clone())?;
            child_b.start_benchmark(b_idx, seed, iterations, num_samples, aux_metrics.clone())?;

            let expected_duration = match loop_mode {
                LoopMode::Samples(samples) => TIME_SLICE * samples as u32,
                LoopMode::Time(duration) => duration,
            };

            // Wait for children to finish, handling time-budget and progress reporting
            let time_start = Instant::now();
            while !commpage.is_some_done() {
                let duration = time_start.elapsed();
                // Monitoring execution time and stopping benchmark on a timer
                if let LoopMode::Time(required_duration) = loop_mode {
                    if duration >= required_duration {
                        break;
                    }
                }

                // Reporting progress to CLI
                if progress_report {
                    let progress = match loop_mode {
                        LoopMode::Time(required_duration) => BenchmarkProgress::SamplingTime {
                            duration,
                            required_duration,
                        },
                        LoopMode::Samples(samples_total) => BenchmarkProgress::SamplingNo {
                            samples_total,
                            sample_no: commpage
                                .load_cursor_value(Role::Candidate)
                                .min(commpage.load_cursor_value(Role::Baseline)),
                        },
                    };
                    if expected_duration > PROGRESS_REPORTING_THRESHOLD
                        || time_start.elapsed() > PROGRESS_REPORTING_THRESHOLD
                    {
                        reporting::report_progress(func_name, progress);
                    }
                }

                sleep(PROGRESS_UPDATE_INTERVAL);
            }
            // Signaling children to exit
            commpage.set_stop();
            let wall_time = time_start.elapsed();

            // Both children are done — read their RPC responses
            let RunBenchmarkResult {
                samples: c_samples,
                aux_metrics: c_aux_metrics,
            } = child_c
                .finish_benchmark()
                .context("Candidate benchmark failed")?;
            let RunBenchmarkResult {
                samples: b_samples,
                aux_metrics: b_aux_metrics,
            } = child_b
                .finish_benchmark()
                .context("Baseline benchmark failed")?;

            // println!("b: {:?}", b_samples);
            // println!("c: {:?}", c_samples);

            let samples = b_samples.into_iter().zip(c_samples).collect::<Vec<_>>();

            let run_result =
                calculate_run_result(&samples, iterations, filter_outliers, noise_threshold)
                    .ok_or(Error::NoMeasurements)
                    .with_context(|| {
                        format!("Unable to calculate results for a benchmark: {}", func_name)
                    })?;

            if let Some(path) = &path_to_dump {
                let file_name = format!("{}.csv", func_name.replace('/', "-"));
                let file_path = path.join(file_name);
                write_csv(&file_path, samples, iterations)
                    .context("Unable to write raw measurements")?;
                sample_dumps.push(file_path);
            }

            if run_result.diff_estimate.significant || !significant_only {
                reporter.benchmark_finished(
                    func_name,
                    &run_result,
                    (&aux_metrics, &b_aux_metrics, &c_aux_metrics),
                );
            }

            // Reporting warning if system time is >5%
            if !quiet {
                check_system_time_bias(
                    &aux_metrics,
                    func_name,
                    wall_time,
                    c_aux_metrics,
                    b_aux_metrics,
                );
            }

            if run_result.diff_estimate.significant && run_result.diff_estimate.pct > 0. {
                exit_code = ExitCode::FAILURE;
                if fail_fast {
                    break;
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
    fn check_system_time_bias(
        aux_metrics: &[String],
        func_name: &str,
        wall_time: Duration,
        c_aux_metrics: Vec<u64>,
        b_aux_metrics: Vec<u64>,
    ) {
        let sys_time_idx = aux_metrics
            .iter()
            .position(|name| name == CPU_SYSTEM_TIME_ID);
        if let Some(system_time_idx) = sys_time_idx {
            let c_sys_time = Duration::from_nanos(c_aux_metrics[system_time_idx]);
            let b_sys_time = Duration::from_nanos(b_aux_metrics[system_time_idx]);

            if c_sys_time * 20 > wall_time || b_sys_time * 20 > wall_time {
                report_system_time_bias(func_name, wall_time, b_sys_time, c_sys_time);
            }
        }
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
        RunResult, Summary,
    };
    use colored::Colorize;
    use std::{
        io::{self, stderr, Write},
        time::Duration,
    };

    pub(super) enum BenchmarkProgress {
        #[allow(dead_code)]
        SamplingNo {
            sample_no: usize,
            samples_total: usize,
        },
        SamplingTime {
            /// Duration benchmark is already running at the moment
            duration: Duration,
            /// How long benchmark should be running
            required_duration: Duration,
        },
    }

    fn clear_progress_line() {
        eprint!("\r\x1b[2K");
        let _ = io::stderr().flush();
    }

    pub(super) trait Reporter {
        fn benchmark_finished(
            &self,
            name: &str,
            results: &RunResult,
            aux_metrics: (&[String], &[u64], &[u64]),
        );
    }

    pub(super) struct VerboseReporter;

    impl Reporter for VerboseReporter {
        fn benchmark_finished(
            &self,
            name: &str,
            results: &RunResult,
            aux_metrics: (&[String], &[u64], &[u64]),
        ) {
            clear_progress_line();

            let base = results.baseline;
            let candidate = results.candidate;

            let significant = results.diff_estimate.significant;

            println!(
                "{}  (n: {}, outliers: {})",
                name.bold(),
                results.diff.n,
                results.outliers
            );

            println!(
                "    {:12}   {:>15} {:>15} {:>15}",
                "",
                "baseline".bold(),
                "candidate".bold(),
                "\u{2206}".bold(),
            );
            println!("    {:12} \u{256d}{}", "", "\u{2500}".repeat(48));
            println!(
                "    {:12} \u{2502} {:>15} {:>15} {:>15}  {:>7}{}{}",
                "mean",
                HumanTime(base.mean),
                HumanTime(candidate.mean),
                colorize(
                    format!("{}", HumanTime(results.diff.mean)),
                    significant,
                    results.diff.mean < 0.
                ),
                colorize(
                    format!("{:+.2}", results.diff_estimate.pct),
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

            // Printing AUX metrics
            let (names, b_aux, c_aux) = aux_metrics;
            for ((name, b), c) in names.iter().zip(b_aux.iter()).zip(c_aux.iter()) {
                println!(
                    "{:>16} \u{2502} {:>15} {:>15} {:>15}  {:>+7.2}%",
                    name,
                    b,
                    c,
                    *c as i64 - *b as i64,
                    (*c as f64 - *b as f64) / (*b as f64) * 100.0
                );
            }
            println!();
        }
    }

    pub(super) struct DefaultReporter;

    impl Reporter for DefaultReporter {
        fn benchmark_finished(
            &self,
            name: &str,
            results: &RunResult,
            _: (&[String], &[u64], &[u64]),
        ) {
            clear_progress_line();

            let base = results.baseline;
            let candidate = results.candidate;
            let diff = results.diff;

            let significant = results.diff_estimate.significant;

            let speedup = results.diff_estimate.pct;
            let candidate_faster = diff.mean < 0.;
            println!(
                "{:50} [ {:>8} ... {:>8} ]    {:>7}{}{}",
                colorize(name, significant, candidate_faster),
                HumanTime(base.mean),
                colorize(
                    format!("{}", HumanTime(candidate.mean)),
                    significant,
                    candidate_faster
                ),
                colorize(format!("{:+.2}", speedup), significant, candidate_faster),
                colorize("%", significant, candidate_faster),
                if significant { "*" } else { "" },
            )
        }
    }

    /// Reporting the progress of a current benchmarking
    ///
    /// Pay attention that frequent reporting might influence the results.
    /// On AWS it was found that request reporting is adding around ~1% of difference
    /// between benchmarks.
    pub(super) fn report_progress(name: &str, progress: BenchmarkProgress) {
        const BAR_WIDTH: usize = 23;
        let progress_line = match progress {
            BenchmarkProgress::SamplingNo {
                sample_no,
                samples_total,
            } => {
                let sample_no = sample_no.min(samples_total);
                let filled = (sample_no * BAR_WIDTH) / samples_total;
                let empty = BAR_WIDTH - filled;
                format!(
                    "\r\x1b[2K{:50} [{}{}] {}/{}",
                    name,
                    "#".repeat(filled),
                    ".".repeat(empty),
                    sample_no,
                    samples_total,
                )
            }
            BenchmarkProgress::SamplingTime {
                duration: loop_time,
                required_duration: total_duration,
            } => {
                let filled = loop_time.as_millis() as usize * BAR_WIDTH
                    / total_duration.as_millis() as usize;
                let empty = BAR_WIDTH - filled;
                format!(
                    "\r\x1b[2K{:50} [{}{}] {:.1}s",
                    name,
                    "#".repeat(filled),
                    ".".repeat(empty),
                    loop_time.as_secs_f32(),
                )
            }
        };
        // Explicitly using write_all() imstead of eprint!(), because latter generates a lot
        // of write() syscalls, which is influencing fairness on Linux
        let mut out = stderr();
        let _ = out.write_all(progress_line.as_bytes());
        let _ = out.flush();
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

    pub(super) fn report_system_time_bias(
        name: &str,
        wall_time: Duration,
        b_sys_time: Duration,
        c_sys_time: Duration,
    ) {
        eprintln!(
            "{}: {} benchmark spent too much time in system mode. Results may be inaccurate",
            "WARN".red(),
            name,
        );
        eprintln!("{:>20}: {:?}", "wall time", wall_time);
        eprintln!("{:>20}: {:?}", "candidate sys", c_sys_time);
        eprintln!("{:>20}: {:?}", "baseline sys", b_sys_time);
    }
}

fn colorize(value: impl Into<String>, do_paint: bool, is_improved: bool) -> impl Display {
    use colored::Colorize;

    let value = value.into();
    if do_paint {
        if is_improved {
            value.green()
        } else {
            value.red()
        }
    } else {
        value.normal()
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

    #[test]
    fn colorize_output() {
        assert_eq!(
            format!("{}", colorize("1.02", true, true)),
            "\u{1b}[32m1.02\u{1b}[0m"
        );
    }
}
