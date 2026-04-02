use regex::Regex;
use std::process::{Command, ExitStatus};

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
    Cmd::run(
        SLEEP_100,
        &[
            "--color",
            "never",
            "compare",
            SLEEP_10,
            "--noise-threshold",
            "99999",
        ],
    )
    .assert_success();
}

/// Run sleep_100 (100ms) as candidate against sleep_10 (10ms) as baseline.
/// With default noise threshold (0.5%), the ~900% regression should be detected.
#[test]
fn regression_detected_with_default_noise_threshold() {
    Cmd::run(SLEEP_100, &["--color", "never", "compare", SLEEP_10])
        .assert_failure()
        .assert_stdout_match("sleep {..} [ {..} ... {..} ] {..} +{..}%*\n");
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
            self.status.code(),
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
        let parts = pattern.split("{..}").collect::<Vec<_>>();
        let escaped: Vec<String> = parts.iter().map(|p| regex::escape(p)).collect();
        let re_pattern = format!("(?s)^{}$", escaped.join(".+"));
        let re = Regex::new(&re_pattern).expect("Invalid regex");
        assert!(
            re.is_match(&self.stdout),
            "Expected stdout to match: {}\nstdout: {}",
            pattern.trim(),
            self.stdout,
        );
        self
    }
}
