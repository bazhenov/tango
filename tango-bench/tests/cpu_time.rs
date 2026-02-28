use std::{hint::black_box, thread, time::Duration};
use tango_bench::{metrics::CpuTime, Metric};

/// Verify that CpuTime measures CPU consumption, not elapsed wall time.
///
/// A thread sleeping for 50ms should consume near-zero CPU time, while
/// a busy loop should register significant CPU time. This test is placed
/// in a separate file (like `rusage.rs`) to run in its own process.
#[test]
fn cpu_time_excludes_sleep() {
    let sleep_cpu = CpuTime::measure_fn(|| {
        thread::sleep(Duration::from_millis(50));
    });

    let busy_cpu = CpuTime::measure_fn(|| {
        let mut sum = 0u64;
        for i in 0..10_000_000 {
            sum = sum.wrapping_add(i);
        }
        black_box(sum);
    });

    assert!(
        busy_cpu > sleep_cpu * 10,
        "CPU time during busy work ({busy_cpu} ns) should far exceed CPU time during sleep ({sleep_cpu} ns)"
    );
}
