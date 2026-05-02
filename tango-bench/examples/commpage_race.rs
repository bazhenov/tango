//! Minimal example: two child processes race a factorial workload,
//! synchronize via a shared Commpage, collect timings locally, and the
//! master prints the percentage difference.

use std::{
    env,
    hint::black_box,
    io::{BufRead, BufReader},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use tango_bench::commpage::{Commpage, Role};

const ITERATIONS: u64 = 100;

/// Simple factorial (wrapping to avoid overflow panics).
#[inline(never)]
fn factorial(n: u64) -> u64 {
    let mut result: u64 = 1;
    for i in 1..=n {
        result = result.wrapping_mul(i);
    }
    result
}

/// Entry point for a child worker process.
fn run_child(shmem_id: &str, role: Role) {
    let commpage = Commpage::open(shmem_id).expect("failed to open commpage");
    let mut samples = Vec::new();

    for i in 0..ITERATIONS {
        let start = Instant::now();
        for _ in 0..black_box(1000000) {
            black_box(factorial(black_box(20)));
        }
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        samples.push(elapsed_ns);

        commpage.advance_cursor(role, i);
        if !commpage.wait_for_cursor_value(role.peer(), i) {
            break;
        }
    }
    commpage.mark_done(role);

    // Print samples to stdout, one per line
    for s in &samples {
        println!("{s}");
    }
}

/// Entry point for the master process.
fn run_master() {
    let commpage = Commpage::create().expect("failed to create commpage");
    let shmem_id = commpage.os_id().to_string();

    let exe = env::current_exe().expect("cannot determine own path");

    // Spawn two children with piped stdout so we can capture their samples.
    let child_c = Command::new(&exe)
        .args(["--child", &shmem_id, "candidate"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn candidate");

    let child_b = Command::new(&exe)
        .args(["--child", &shmem_id, "baseline"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn baseline");

    // Wait for children and read their samples from stdout.
    let output_c = child_c.wait_with_output().expect("candidate failed");
    let output_b = child_b.wait_with_output().expect("baseline failed");

    let parse_samples = |output: Vec<u8>| -> Vec<u64> {
        BufReader::new(output.as_slice())
            .lines()
            .map_while(Result::ok)
            .filter(|l| !l.is_empty())
            .map(|l| l.parse().expect("invalid sample"))
            .collect()
    };

    let samples_c = parse_samples(output_c.stdout);
    let samples_b = parse_samples(output_b.stdout);

    let samples: Vec<(u64, u64)> = samples_c.into_iter().zip(samples_b).collect();

    for (c, b) in &samples {
        let diff = (*c as f64 - *b as f64) / (*b as f64);
        println!(
            "{:5.1}% ({} ms)",
            diff * 100.0,
            Duration::from_nanos(*b).as_millis()
        );
    }

    let sum: i64 = samples.iter().map(|(c, b)| *c as i64 - *b as i64).sum();
    println!("Mean: {:.0} us", sum / samples.len() as i64 / 1_000);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() >= 4 && args[1] == "--child" {
        let shmem_id = &args[2];
        let role = match args[3].as_str() {
            "candidate" => Role::Candidate,
            "baseline" => Role::Baseline,
            other => panic!("unknown role: {other}"),
        };
        run_child(shmem_id, role);
    } else {
        run_master();
    }
}
