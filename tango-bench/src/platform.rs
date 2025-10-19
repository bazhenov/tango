use std::{ops::Sub, time::Duration};

/// Reexports
pub use active_platform::rusage;

#[derive(Debug)]
pub struct RUsage {
    pub user_time: Duration,
    pub system_time: Duration,
}

impl Sub for RUsage {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            user_time: self.user_time - other.user_time,
            system_time: self.system_time - other.system_time,
        }
    }
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

    pub fn rusage() -> RUsage {
        use libc::{getrusage, rusage, RUSAGE_SELF};

        let mut usage = unsafe { MaybeUninit::<rusage>::zeroed().assume_init() };
        unsafe { getrusage(RUSAGE_SELF, &mut usage as *mut _) };

        usage.into()
    }

    impl From<libc::rusage> for RUsage {
        fn from(usage: libc::rusage) -> Self {
            fn timeval_to_duration(tv: libc::timeval) -> Duration {
                Duration::from_secs(tv.tv_sec as u64) + Duration::from_micros(tv.tv_usec as u64)
            }
            Self {
                user_time: timeval_to_duration(usage.ru_utime),
                system_time: timeval_to_duration(usage.ru_stime),
            }
        }
    }
}

#[cfg(target_os = "windows")]
pub mod windows {
    use super::*;
    use ::windows::Win32::{
        Foundation::FILETIME,
        System::Threading::{GetCurrentProcess, GetProcessTimes},
    };

    pub fn rusage() -> RUsage {
        let mut dummy = FILETIME::default();
        let mut kernel_time = FILETIME::default();
        let mut user_time = FILETIME::default();
        unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut dummy as *mut _,
                &mut dummy as *mut _,
                &mut kernel_time as *mut _,
                &mut user_time as *mut _,
            )
        }
        .unwrap();

        RUsage {
            user_time: filetime_to_duration(user_time),
            system_time: filetime_to_duration(kernel_time),
        }
    }

    fn filetime_to_duration(time: FILETIME) -> Duration {
        // FILETIME is expressed in 100ns time units
        Duration::from_micros(time.dwLowDateTime as u64 / 10)
    }
}
