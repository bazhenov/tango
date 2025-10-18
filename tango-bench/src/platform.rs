use std::time::Duration;

pub use active_platform::rusage;

#[derive(Debug)]
pub struct RUsage {
    pub user_time: Duration,
    pub system_time: Duration,
}

#[cfg(target_family = "unix")]
pub use unix as active_platform;

#[cfg(target_family = "unix")]
pub mod unix {
    use super::*;
    use std::{mem::MaybeUninit, time::Duration};

    pub fn rusage<T>(f: impl Fn() -> T) -> (T, RUsage) {
        use libc::rusage;

        let mut usage = unsafe {
            let mut usage = MaybeUninit::<rusage>::uninit();
            libc::getrusage(0, usage.as_mut_ptr());
            usage.assume_init()
        };

        let utime_before = usage.ru_utime;
        let stime_before = usage.ru_stime;

        let result = f();

        unsafe { libc::getrusage(0, &mut usage as *mut rusage) };
        let utime_after = usage.ru_utime;
        let stime_after = usage.ru_stime;

        let user_time = Duration::from_secs((utime_after.tv_sec - utime_before.tv_sec) as u64)
            + Duration::from_millis((utime_after.tv_usec - utime_before.tv_usec) as u64);

        let system_time = Duration::from_secs((stime_after.tv_sec - stime_before.tv_sec) as u64)
            + Duration::from_millis((stime_after.tv_usec - stime_before.tv_usec) as u64);

        let rusage = RUsage {
            user_time,
            system_time,
        };
        (result, rusage)
    }
}
