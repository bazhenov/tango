//! Shared memory commpage for lock-step synchronization.
//!
//! The commpage is a single shared memory region accessible to all three processes
//! (Runner, Candidate, Baseline). It contains two cursors -- one per child -- used
//! to synchronize lock-step execution. Measurement data flows via JSON-RPC responses,
//! not through the commpage.

use shared_memory::{Shmem, ShmemConf};
use std::{
    mem::size_of,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};
use thiserror::Error;

/// Magic bytes: "TANGOCMP" as a u64
const MAGIC: u64 = 0x54414E47_4F434D50;

/// Protocol version
const VERSION: u32 = 3;

/// Bit 63 of cursor -- set when the child exits the measurement loop
const DONE_BIT: u64 = 1 << 63;

/// Bit 0 of flags -- set by runner to signal early termination
const STOP_FLAG: u32 = 1;

/// The commpage memory layout: header + two cursors laid out contiguously.
///
/// ```text
/// Offset  Size    Field
/// 0x000   8       magic
/// 0x008   4       version
/// 0x00C   4       flags (AtomicU32)
/// 0x010   8       cursor_c (AtomicU64)
/// 0x018   8       cursor_b (AtomicU64)
/// ```
#[repr(C)]
struct CommpageLayout {
    magic: u64,
    version: u32,
    flags: AtomicU32,
    cursor_c: Cursor,
    cursor_b: Cursor,
}

#[repr(C)]
struct Cursor(AtomicU64);

impl Cursor {
    fn set_value(&mut self, value: u64) {
        self.0.store(value & !DONE_BIT, Ordering::Relaxed);
    }

    fn load(&self) -> (bool, u64) {
        let val = self.0.load(Ordering::Acquire);
        (val & DONE_BIT != 0, val & !DONE_BIT)
    }

    fn set_done(&mut self) {
        self.0.fetch_or(DONE_BIT, Ordering::AcqRel);
    }

    fn reset(&mut self) {
        self.0.store(0, Ordering::Relaxed);
    }
}

impl CommpageLayout {
    fn init(&mut self) {
        self.magic = MAGIC;
        self.version = VERSION;
        *self.flags.get_mut() = 0;
        *self.cursor_c.0.get_mut() = 0;
        *self.cursor_b.0.get_mut() = 0;
    }

    fn validate(&self) -> Result<(), CommpageError> {
        if self.magic != MAGIC {
            return Err(CommpageError::BadMagic);
        }
        if self.version != VERSION {
            return Err(CommpageError::BadVersion {
                got: self.version,
                expected: VERSION,
            });
        }
        Ok(())
    }

    fn cursor(&self, role: Role) -> &Cursor {
        match role {
            Role::Candidate => &self.cursor_c,
            Role::Baseline => &self.cursor_b,
        }
    }

    fn cursor_mut(&mut self, role: Role) -> &mut Cursor {
        match role {
            Role::Candidate => &mut self.cursor_c,
            Role::Baseline => &mut self.cursor_b,
        }
    }
}

/// Which role a child process plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Candidate,
    Baseline,
}

impl Role {
    pub fn peer(self) -> Self {
        match self {
            Self::Candidate => Self::Baseline,
            Self::Baseline => Self::Candidate,
        }
    }
}

/// Handle to a commpage backed by shared memory.
pub struct Commpage {
    shmem: Shmem,
}

impl Commpage {
    /// Create a new commpage with a unique OS name. Initializes header and both cursors.
    pub fn create() -> Result<Self, CommpageError> {
        let shmem = ShmemConf::new()
            .size(size_of::<CommpageLayout>())
            .create()
            .map_err(CommpageError::Shmem)?;

        let layout = unsafe { &mut *(shmem.as_ptr() as *mut CommpageLayout) };
        layout.init();

        Ok(Commpage { shmem })
    }

    /// Open an existing commpage by its OS identifier. Validates magic and version.
    pub fn open(name: &str) -> Result<Self, CommpageError> {
        let shmem = ShmemConf::new()
            .os_id(name)
            .open()
            .map_err(CommpageError::Shmem)?;

        let layout = unsafe { &*(shmem.as_ptr() as *const CommpageLayout) };
        layout.validate()?;

        Ok(Commpage { shmem })
    }

    /// Get the OS identifier for this shared memory region.
    pub fn os_id(&self) -> &str {
        self.shmem.get_os_id()
    }

    fn layout(&self) -> &CommpageLayout {
        unsafe { &*(self.shmem.as_ptr() as *const CommpageLayout) }
    }

    fn layout_mut(&mut self) -> &mut CommpageLayout {
        unsafe { &mut *(self.shmem.as_ptr() as *mut CommpageLayout) }
    }

    /// Advance this role's cursor after completing a sample.
    pub fn advance_cursor(&mut self, role: Role, sample_no: u64) {
        self.layout_mut().cursor_mut(role).set_value(sample_no);
    }

    /// Spin-wait until the peer's cursor reaches at least `target`, or the peer sets DONE.
    ///
    /// Returns `true` if the peer caught up, `false` if the peer exited early (DONE).
    pub fn wait_for_cursor_value(&self, role: Role, target: u64) -> bool {
        let cursor = self.layout().cursor(role);
        loop {
            let (stop, val) = cursor.load();
            if stop {
                return false;
            }
            if val >= target {
                return true;
            } else {
                std::hint::spin_loop();
            }
        }
    }

    /// Mark this role's cursor as done.
    pub fn mark_done(&mut self, role: Role) {
        self.layout_mut().cursor_mut(role).set_done();
    }

    /// Check if this role's cursor has the DONE bit set.
    pub fn is_done(&self, role: Role) -> bool {
        let (done, _) = self.layout().cursor(role).load();
        done
    }

    /// Either candidate or baseline has DONE bit set.
    pub fn is_some_done(&self) -> bool {
        self.is_done(Role::Candidate) || self.is_done(Role::Baseline)
    }

    /// Read the sample counter for a given role (cursor value without the DONE bit).
    /// Sample counter is the number of a sample the children is collecting right now. 0 means
    /// that measurement has not start yet.
    pub fn load_sample_counter(&self, role: Role) -> usize {
        let (_, value) = self.layout().cursor(role).load();
        value as usize
    }

    /// Set the STOP flag. Called by runner to signal time-budget exhaustion.
    pub fn set_stop(&mut self) {
        self.layout().flags.fetch_or(STOP_FLAG, Ordering::Release);
    }

    /// Check if the STOP flag is set.
    pub fn is_stopped(&self) -> bool {
        self.layout().flags.load(Ordering::Acquire) & STOP_FLAG != 0
    }

    /// Reset the commpage for a new benchmark run.
    pub fn reset(&mut self) {
        self.layout_mut().flags.store(0, Ordering::Release);
        self.layout_mut().cursor_c.reset();
        self.layout_mut().cursor_b.reset();
    }
}

#[derive(Debug, Error)]
pub enum CommpageError {
    #[error("shared memory error: {0}")]
    Shmem(shared_memory::ShmemError),

    #[error("invalid commpage magic number")]
    BadMagic,

    #[error("unsupported commpage version: {got} (expected {expected})")]
    BadVersion { got: u32, expected: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn create_and_open() {
        let cp = Commpage::create().unwrap();
        let id = cp.os_id().to_string();
        let cp2 = Commpage::open(&id).unwrap();
        assert!(!cp2.is_stopped());
    }

    #[test]
    fn stop_flag() {
        let mut cp = Commpage::create().unwrap();
        assert!(!cp.is_stopped());
        cp.set_stop();
        assert!(cp.is_stopped());
    }

    #[test]
    fn done_bit() {
        let mut cp = Commpage::create().unwrap();
        assert!(!cp.is_done(Role::Candidate));
        cp.advance_cursor(Role::Candidate, 1);
        cp.mark_done(Role::Candidate);

        assert!(cp.is_done(Role::Candidate));
        assert_eq!(cp.load_sample_counter(Role::Candidate), 1);
    }

    #[test]
    fn peer_synchronization_two_threads() {
        let mut cp = Commpage::create().unwrap();
        let id = cp.os_id().to_string();

        let num_samples = 64u64;

        let handle = thread::spawn(move || {
            let mut cp = Commpage::open(&id).unwrap();

            for i in 0..num_samples {
                cp.advance_cursor(Role::Candidate, i);
                cp.wait_for_cursor_value(Role::Baseline, i);
            }
            cp.mark_done(Role::Candidate);
        });

        for i in 0..num_samples {
            cp.advance_cursor(Role::Baseline, i);
            cp.wait_for_cursor_value(Role::Candidate, i);
        }
        cp.mark_done(Role::Baseline);

        handle.join().unwrap();

        assert_eq!(
            cp.load_sample_counter(Role::Baseline),
            (num_samples - 1) as usize
        );
        assert_eq!(
            cp.load_sample_counter(Role::Candidate),
            (num_samples - 1) as usize
        );
    }
}
