use regex::Regex;
use std::{
    fs,
    process::{Command, ExitStatus},
};
use temp_dir::TempDir;

const SLEEP_10: &str = env!("CARGO_BIN_EXE_sleep_10");
const MULTIPLE: &str = env!("CARGO_BIN_EXE_multiple");
const SLEEP_100: &str = env!("CARGO_BIN_EXE_sleep_100");
const NOT_A_BENCH: &str = env!("CARGO_BIN_EXE_not_a_bench");
const SLEEP_PANIC: &str = env!("CARGO_BIN_EXE_sleep_panic");

#[test]
fn help() {
    Cmd::run(SLEEP_10, &["help"])
        .assert_success()
        .assert_stdout_contains("Tango benchmarking harness")
        .assert_stdout_contains("Usage: {..} [OPTIONS] [COMMAND]")
        .assert_stdout_contains("Options:")
        .assert_stdout_contains("Commands:")
        .assert_stdout_contains("compare{..}Run paired benchmarking to compare two executables");
}

#[test]
fn compare() {
    Cmd::run(SLEEP_10, &["compare"])
        .assert_success()
        .assert_stdout_match("sleep {..} [ {..} ... {..} ]{..}\n");
}

#[test]
fn not_existent_benchmark() {
    Cmd::run(SLEEP_10, &["compare", "not-existent"])
        .assert_failure()
        .assert_stderr_contains("Benchmark not found: {..}");
}

#[test]
fn not_a_benchmark() {
    Cmd::run(SLEEP_10, &["compare", NOT_A_BENCH])
        .assert_failure()
        .assert_stderr_contains("Failed to list benchmarks (baseline)");
}

#[test]
fn solo() {
    Cmd::run(SLEEP_10, &["solo"])
        .assert_success()
        .assert_stdout_match("sleep {..} [ {..} ... {..} ... {..} ]  stddev: {..}\n");
}

/// Run sleep_100 (100ms) as candidate against sleep_10 (10ms) as baseline.
/// The candidate is slower, so this is a regression. A high noise threshold should suppress it.
#[test]
fn noise_threshold_mutes_regression() {
    Cmd::run(
        SLEEP_100,
        &["compare", SLEEP_10, "--noise-threshold", "99999"],
    )
    .assert_success()
    .assert_stdout_match("sleep {..} [ {..} ... {..} ]{..} +{..}%\n");
}

/// Run sleep_100 (100ms) as candidate against sleep_10 (10ms) as baseline.
/// With default noise threshold (0.5%), the ~900% regression should be detected.
#[test]
fn regression_detected_with_default_noise_threshold() {
    Cmd::run(SLEEP_100, &["compare", SLEEP_10])
        .assert_failure()
        .assert_stdout_match("sleep {..} [ {..} ... {..} ]{..} +{..}%*\n");
}

#[test]
fn benchmark_with_panic() {
    Cmd::run(SLEEP_10, &["compare", SLEEP_PANIC])
        .assert_failure()
        .assert_stderr_contains("Intended panic");
}

#[test]
fn benchmark_with_multiple_tests() {
    Cmd::run(MULTIPLE, &["compare", "--noise-threshold", "100", MULTIPLE])
        .assert_success()
        .assert_stdout_contains("bench1 {..} [ {..} ... {..} ]")
        .assert_stdout_contains("bench2 {..} [ {..} ... {..} ]");
}

fn gnuplot_available() -> bool {
    Command::new("gnuplot")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// When passed --gnuplot and -d, the benchmark harness should produce SVG plot files.
#[test]
fn gnuplot_produces_svg() {
    if !gnuplot_available() {
        eprintln!("skipping test: gnuplot not installed");
        return;
    }
    let tmp_dir = TempDir::new().unwrap();

    let dump_path = tmp_dir.path().to_str().unwrap();
    Cmd::run(SLEEP_10, &["compare", "-d", dump_path, "--gnuplot"]).assert_success();

    let all_files = fs::read_dir(&tmp_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect::<Vec<_>>();

    // Verify at least one SVG file was produced
    let svg_files: Vec<_> = all_files
        .iter()
        .filter(|e| e.extension().is_some_and(|ext| ext == "svg"))
        .collect();

    assert!(
        !svg_files.is_empty(),
        "Expected at least one SVG file in {}, found none. Contents: {:?}",
        dump_path,
        &all_files,
    );

    // Verify the SVG file is valid (starts with SVG content)
    let svg_path = svg_files[0];
    let svg_content = fs::read_to_string(svg_path).expect("should be able to read SVG file");
    assert!(
        svg_content.contains("<svg"),
        "SVG file {} does not contain <svg tag. Content starts with: {}",
        svg_path.display(),
        &svg_content[..svg_content.len().min(200)],
    );
}

struct Cmd {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}

impl Cmd {
    fn run(cmd: &str, args: &[&str]) -> Self {
        let output = Command::new(cmd)
            .args(args)
            .output()
            .expect("failed to execute");

        Cmd {
            stdout: String::from_utf8(output.stdout).unwrap(),
            stderr: String::from_utf8(output.stderr).unwrap(),
            status: output.status,
        }
    }

    fn assert_success(&self) -> &Self {
        assert!(
            self.status.success(),
            "Expected exit code 0, got {:?}\nstdout: {}\nstderr: {}",
            self.status,
            self.stdout,
            self.stderr,
        );
        self
    }

    fn assert_failure(&self) -> &Self {
        assert!(
            !self.status.success(),
            "Expected non-zero exit code\nstdout: {}\nstderr: {}",
            self.stdout,
            self.stderr,
        );
        self
    }

    fn assert_stdout_match(&self, pattern: &str) -> &Self {
        let re_pattern = format!("(?s)^{}$", compile_pattern(pattern));
        let re = Regex::new(&re_pattern).expect("Invalid regex");

        assert!(
            re.is_match(&self.stdout),
            "Expected stdout to match: {}\nstdout: {}",
            pattern.trim(),
            self.stdout,
        );
        self
    }

    fn assert_stdout_contains(&self, pattern: &str) -> &Self {
        let re_pattern = format!("(?s){}", compile_pattern(pattern));
        let re = Regex::new(&re_pattern).expect("Invalid regex");

        assert!(
            re.find(&self.stdout).is_some(),
            "Expected stdout to contain: {}\nstdout: {}",
            pattern,
            self.stdout,
        );
        self
    }

    fn assert_stderr_contains(&self, pattern: &str) -> &Self {
        let re_pattern = format!("(?si){}", compile_pattern(pattern));
        let re = Regex::new(&re_pattern).expect("Invalid regex");

        assert!(
            re.find(&self.stderr).is_some(),
            "Expected stderr to contain: {}\nstderr: {}",
            pattern,
            self.stderr,
        );
        self
    }
}

fn compile_pattern(pattern: &str) -> String {
    let parts = pattern.split("{..}").collect::<Vec<_>>();
    let escaped = parts.into_iter().map(regex::escape).collect::<Vec<_>>();
    escaped.join(".+")
}
