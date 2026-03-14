use libc::{exit, fork};
use shared_memory::ShmemConf;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

fn main() {
    let shmem = ShmemConf::new().size(4096).create().unwrap();
    let os_name = shmem.get_os_id();
    let pid = unsafe { fork() };
    let shmem = if pid == 0 {
        ShmemConf::new().os_id(os_name).open().unwrap()
    } else {
        shmem
    };

    let shared = unsafe { &*(shmem.as_ptr() as *const AtomicU64) };
    if shmem.is_owner() {
        let start = Instant::now();
        let mut value = 0;
        while Instant::now().duration_since(start) < Duration::from_secs(10) && value == 0 {
            value = shared.load(Ordering::Acquire);
        }
        assert_eq!(value, 42);
    } else {
        shared.store(42, Ordering::Release);
        unsafe { exit(0) };
    }
}
