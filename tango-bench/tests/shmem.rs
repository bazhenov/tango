use shared_memory::ShmemConf;
use std::{
    env,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

fn main() {
    let (shmem, child) = if let Ok(shmem_name) = env::var("SHMEM") {
        let shmem = ShmemConf::new().os_id(shmem_name).open().unwrap();
        println!("Creating shared memory: {}", shmem.get_os_id());
        (shmem, None)
    } else {
        let shmem = ShmemConf::new().size(4096).create().unwrap();

        let child = Command::new(env::current_exe().unwrap())
            .env("SHMEM", shmem.get_os_id())
            .spawn()
            .unwrap();

        (shmem, Some(child))
    };

    let shared = unsafe { &*(shmem.as_ptr() as *const AtomicU64) };
    if let Some(child) = child {
        let start = Instant::now();
        let mut value = 0;
        while Instant::now().duration_since(start) < Duration::from_secs(10) && value == 0 {
            value = shared.load(Ordering::Acquire);
        }
        assert_eq!(value, 42);
        child.wait_with_output().unwrap();
        println!("Ok");
    } else {
        shared.store(42, Ordering::Release);
    }
}
