use std::{
    thread,
    time::{Duration, Instant},
};
use tango_bench::platform;

/// The idea of the test is that we keep spawning and joining threads that do nothing for a short while.
/// Since the threads do not perform any meaningful computations, most of the time should be spent in the kernel;
/// hence, system time should be greater than user time. We run this for a specified duration because the system timer
/// used by `getrusage()` has platform-specific resolution. Empirically, 100ms is sufficient on macOS and Linux.
///
/// This test is intentionally placed in a separate file because it needs to be executed in a single-threaded
/// environment to remain stable. When included in unit tests (which run in parallel), it becomes flaky, because
/// `getrusage()` reports resource usage for the whole process.
#[test]
fn check_rusage() {
    #[cfg(not(target_os = "windows"))]
    const TEST_DURATION: Duration = Duration::from_millis(100);
    /// On Windows `GetProcessTimes()` has a precision of 1/64s.,
    /// so 100ms is not enough to get reliable reading from the OS
    #[cfg(target_os = "windows")]
    const TEST_DURATION: Duration = Duration::from_millis(1000);

    let start_ts = Instant::now();
    let (_, rusage) = platform::rusage(|| {
        while Instant::now() - start_ts < TEST_DURATION {
            thread::spawn(|| {}).join().unwrap();
        }
    });
    assert!(rusage.system_time > rusage.user_time,
            "Overhead of thread spawning (system time: {:?}) should be higher than cost of computations (user time: {:?})",
            rusage.system_time, rusage.user_time);
}
