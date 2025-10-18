use std::time::Duration;

/// Reexports
pub use active_platform::rusage;

#[derive(Debug)]
pub struct RUsage {
    pub user_time: Duration,
    pub system_time: Duration,
}

#[cfg(target_os = "macos")]
pub use macos as active_platform;

#[cfg(target_os = "linux")]
pub use linux as active_platform;

#[cfg(target_os = "linux")]
pub mod linux {
    use super::*;
    use std::{mem::MaybeUninit, time::Duration};

    pub fn rusage<T>(f: impl Fn() -> T) -> (T, RUsage) {
        use libc::{rusage, RUSAGE_SELF};

        let mut usage = unsafe {
            let mut usage = MaybeUninit::<rusage>::uninit();
            libc::getrusage(RUSAGE_SELF, usage.as_mut_ptr());
            usage.assume_init()
        };

        let utime_before = usage.ru_utime;
        let stime_before = usage.ru_stime;

        let result = f();

        unsafe { libc::getrusage(RUSAGE_SELF, &mut usage as *mut rusage) };
        let utime_after = usage.ru_utime;
        let stime_after = usage.ru_stime;

        let user_time = Duration::from_secs((utime_after.tv_sec - utime_before.tv_sec) as u64)
            + Duration::from_micros((utime_after.tv_usec - utime_before.tv_usec) as u64);

        let system_time = Duration::from_secs((stime_after.tv_sec - stime_before.tv_sec) as u64)
            + Duration::from_micros((stime_after.tv_usec - stime_before.tv_usec) as u64);

        let rusage = RUsage {
            user_time,
            system_time,
        };
        (result, rusage)
    }
}

pub mod macos {
    use crate::platform::RUsage;
    use std::time::Duration;

    pub fn rusage<T>(f: impl Fn() -> T) -> (T, RUsage) {
        (
            f(),
            RUsage {
                user_time: Duration::from_millis(0),
                system_time: Duration::from_millis(0),
            },
        )
    }
}
