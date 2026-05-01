//! Minimal example: two child processes race a factorial workload,
//! report per-iteration timings through a shared Commpage, and the
//! master prints the percentage difference.

use std::{
    env,
    hint::black_box,
    process::Command,
    thread,
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
    let lane = commpage.get_lane(role);
    let mut samples_written = 0;

    for i in 0..ITERATIONS {
        let start = Instant::now();
        for _ in 0..black_box(1000000) {
            black_box(factorial(black_box(20)));
        }
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        lane.push_sample(i, elapsed_ns);
        samples_written += 1;

        if !commpage.peer_lane(role).wait_for_cursor(samples_written) {
            break;
        }
    }
    lane.mark_done();
}

/// Entry point for the master process.
fn run_master() {
    let commpage = Commpage::create().expect("failed to create commpage");
    let shmem_id = commpage.os_id().to_string();

    let exe = env::current_exe().expect("cannot determine own path");

    // Spawn two children: candidate and baseline.
    let mut child_c = Command::new(&exe)
        .args(["--child", &shmem_id, "candidate"])
        .spawn()
        .expect("failed to spawn candidate");

    let mut child_b = Command::new(&exe)
        .args(["--child", &shmem_id, "baseline"])
        .spawn()
        .expect("failed to spawn baseline");

    // Drain all samples after children are done.
    let mut samples_c = Vec::new();
    let mut samples_b = Vec::new();

    let lane_c = commpage.lane_c();
    let lane_b = commpage.lane_b();

    while !lane_c.is_done() || !lane_b.is_done() {
        if !lane_c.is_done() {
            lane_c.drain_samples(&mut samples_c).unwrap();
        }
        if !lane_b.is_done() {
            lane_b.drain_samples(&mut samples_b).unwrap();
        }
        println!("B: {}, C: {}", samples_b.len(), samples_c.len());
        thread::sleep(Duration::from_millis(100));
    }

    // Wait for both children to finish.
    child_c.wait().expect("candidate process failed");
    child_b.wait().expect("baseline process failed");

    // Compute median elapsed time for each lane.
    // println!("{:?}", samples_b);
    // println!("{:?}", samples_c);

    let samples = samples_c
        .iter()
        .copied()
        .zip(samples_b.iter().copied())
        .collect::<Vec<_>>();

    for (c, b) in samples.iter().copied() {
        let diff = (c as f64 - b as f64) / (b as f64);
        println!(
            "{:5.1}% ({} ms)",
            diff * 100.0,
            Duration::from_nanos(b).as_millis()
        );
    }

    let sum = samples
        .iter()
        .copied()
        .map(|(c, b)| c as i64 - b as i64)
        .sum::<i64>();
    println!("Mean: {:.0} us", sum / samples.len() as i64 / 1_000);
}

// fn median(v: &mut [f64]) -> f64 {
//     // Filter out MISSED_SAMPLE sentinels.
//     let mut clean: Vec<f64> = v.iter().copied().collect();
//     assert!(!clean.is_empty(), "no valid samples collected");
//     clean.sort_unstable();
//     let mid = clean.len() / 2;
//     if clean.len().is_multiple_of(2) {
//         (clean[mid - 1] + clean[mid]) as f64 / 2.0
//     } else {
//         clean[mid] as f64
//     }
// }

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
