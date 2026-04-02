use regex::Regex;
use std::process::{Command, ExitStatus, Output};

const SLEEP_10: &str = env!("CARGO_BIN_EXE_sleep_10");
const SLEEP_100: &str = env!("CARGO_BIN_EXE_sleep_100");

#[test]
fn cli_tests() {
    trycmd::TestCases::new()
        .case("tests/cmd/*.toml")
        .default_bin_name("sleep_10")
        .run();
}

/// Run sleep_100 (100ms) as candidate against sleep_10 (10ms) as baseline.
/// The candidate is slower, so this is a regression. A high noise threshold should suppress it.
#[test]
fn noise_threshold_mutes_regression() {
    let (status, stdout, stderr) = execute(
        SLEEP_100,
        &[
            "--color",
            "never",
            "compare",
            SLEEP_10,
            "--noise-threshold",
            "99999",
        ],
    );

    assert!(
        status.success(),
        "Expected exit code 0 with high noise threshold, got {:?}\nstdout: {}\nstderr: {}",
        status.code(),
        stdout,
        stderr,
    );
}

/// Run sleep_100 (100ms) as candidate against sleep_10 (10ms) as baseline.
/// With default noise threshold (0.5%), the ~900% regression should be detected.
#[test]
fn regression_detected_with_default_noise_threshold() {
    let (status, stdout, stderr) = execute(SLEEP_100, &["--color", "never", "compare", SLEEP_10]);

    let pattern =
        Regex::new(r"sleep[ ]+\[ .+ \.\.\. .+ \][ ]+\+[0-9\.]+%\*\n").expect("Invalid regex");

    assert!(
        !status.success(),
        "Expected non-zero exit code with default noise threshold\nstdout: {}\nstderr: {}",
        stdout,
        stderr,
    );
    assert!(
        pattern.is_match(&stdout),
        "Expected non-zero exit code with default noise threshold\nstdout: {}\nstderr: {}",
        stdout,
        stderr,
    );
}

fn execute(cmd: &str, args: &[&str]) -> (ExitStatus, String, String) {
    let output = Command::new(SLEEP_100)
        .args(args)
        .output()
        .expect("failed to execute");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    (output.status, stdout, stderr)
}
