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

#[cfg(target_os = "windows")]
pub use windows as active_platform;

#[cfg(target_os = "linux")]
pub mod linux {
    pub use super::unix::rusage;
}

#[cfg(target_os = "macos")]
pub mod macos {
    pub use super::unix::rusage;
}

#[cfg(target_family = "unix")]
pub mod unix {
    use super::RUsage;
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

#[cfg(target_os = "windows")]
pub mod windows {
    use super::*;
    use ::windows::Win32::{
        Foundation::FILETIME,
        System::Threading::{GetCurrentProcess, GetProcessTimes},
    };

    pub fn rusage<T>(f: impl Fn() -> T) -> (T, RUsage) {
        let mut dummy = FILETIME::default();
        let mut kernel_time_before = FILETIME::default();
        let mut kernel_time_after = FILETIME::default();
        let mut user_time_before = FILETIME::default();
        let mut user_time_after = FILETIME::default();
        let self_process = unsafe { GetCurrentProcess() };
        unsafe {
            GetProcessTimes(
                self_process,
                &mut dummy as *mut _,
                &mut dummy as *mut _,
                &mut kernel_time_before as *mut _,
                &mut user_time_before as *mut _,
            )
            .unwrap()
        };
        let result = f();
        unsafe {
            GetProcessTimes(
                self_process,
                &mut dummy as *mut _,
                &mut dummy as *mut _,
                &mut kernel_time_after as *mut _,
                &mut user_time_after as *mut _,
            )
            .unwrap()
        };

        let system_time = filetime_to_duration(kernel_time_before, kernel_time_after);
        let user_time = filetime_to_duration(user_time_before, user_time_after);

        (
            result,
            RUsage {
                user_time,
                system_time,
            },
        )
    }

    fn filetime_to_duration(before: FILETIME, after: FILETIME) -> Duration {
        // FILETIME is expressed in 100ns time units
        Duration::from_micros((after.dwLowDateTime - before.dwLowDateTime) as u64 / 10)
    }
}
