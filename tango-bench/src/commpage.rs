//! Shared memory commpage for process-isolated benchmarking.
//!
//! The commpage is a single shared memory region accessible to all three processes
//! (Runner, Candidate, Baseline). It contains two lanes -- one per child. Each child
//! writes to its own lane and reads the peer's write_cursor for synchronization.

use shared_memory::{Shmem, ShmemConf};
use std::{
    mem::size_of,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};
use thiserror::Error;

/// Magic bytes: "TANGOCMP" as a u64
const MAGIC: u64 = 0x54414E47_4F434D50;

/// This value means that sample was missed during commpage draining
pub const MISSED_SAMPLE: u64 = u64::MAX;

/// Protocol version
const VERSION: u32 = 1;

/// Number of sample slots per lane (must be a power of two)
pub const NUM_SLOTS: usize = 128;

/// Bit 63 of write_cursor -- set when the child exits the measurement loop
const DONE_BIT: u64 = 1 << 63;

/// Bit 0 of flags -- set by runner to signal early termination
const STOP_FLAG: u32 = 1;

/// Result of draining samples from a lane.
pub struct DrainResult {
    /// The new read position (equal to current write_cursor).
    pub new_pos: u64,
    /// Number of samples that were overwritten before the reader could drain them.
    pub skipped: u64,
}

/// The commpage memory layout: header + two lanes laid out contiguously.
///
/// ```text
/// Offset  Size    Field
/// 0x000   8       magic
/// 0x008   4       version
/// 0x00C   4       flags (AtomicU32)
/// 0x010   8       write_cursor_c (AtomicU64)
/// 0x018   N*8     samples_c[N]
/// 0x418   8       write_cursor_b (AtomicU64)
/// 0x420   N*8     samples_b[N]
/// ```
#[repr(C)]
struct CommpageLayout {
    magic: u64,
    version: u32,
    flags: AtomicU32,
    lane_c: Lane,
    lane_b: Lane,
}

impl CommpageLayout {
    /// Initialize all fields to their default values.
    fn init(&mut self) {
        self.magic = MAGIC;
        self.version = VERSION;
        *self.flags.get_mut() = 0;
        self.lane_c.init();
        self.lane_b.init();
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
}

/// A single lane in the commpage (write_cursor + ring buffer).
#[repr(C)]
pub struct Lane {
    write_cursor: AtomicU64,
    samples: [AtomicU64; NUM_SLOTS],
}

impl Lane {
    fn init(&mut self) {
        *self.write_cursor.get_mut() = 0;
        for slot in self.samples.iter_mut() {
            *slot.get_mut() = 0;
        }
    }

    /// Read the write_cursor value (without the DONE bit) -- i.e. the sample count.
    pub fn sample_count(&self) -> usize {
        (self.write_cursor.load(Ordering::Acquire) & !DONE_BIT) as usize
    }

    /// Check if the DONE bit is set on this lane's write_cursor.
    pub fn is_done(&self) -> bool {
        self.write_cursor.load(Ordering::Acquire) & DONE_BIT != 0
    }

    /// Write a sample and advance the write_cursor. Called by the child in its measurement loop.
    ///
    /// `sample_no` is the 0-based index of the sample being written.
    /// After this call, write_cursor == sample_no + 1.
    pub fn push_sample(&self, sample_no: u64, elapsed_ns: u64) {
        let slot = (sample_no as usize) & (NUM_SLOTS - 1);
        self.samples[slot].store(elapsed_ns, Ordering::Relaxed);
        // Publish: write_cursor = sample_no + 1
        self.write_cursor.store(sample_no + 1, Ordering::Release);
    }

    /// Spin-wait until this lane's write_cursor reaches at least `target`, or the DONE bit is set.
    ///
    /// Returns `true` if the cursor caught up, `false` if the writer exited early (DONE).
    pub fn wait_for_cursor(&self, target: u64) -> bool {
        loop {
            let cursor = self.write_cursor.load(Ordering::Acquire);
            if cursor & DONE_BIT != 0 {
                return false;
            }
            if cursor >= target {
                return true;
            }
            std::hint::spin_loop();
        }
    }

    /// Mark this lane as done.
    pub fn mark_done(&self) {
        self.write_cursor.fetch_or(DONE_BIT, Ordering::Release);
    }

    /// Drain fresh samples from Lane and add them to a specified [`Vec`].
    /// The size of a [`Vec`] dictates from which position of ring buffer samples are copied.
    /// Returns `Err()` if some samples were skipped (overwritten in the ring buffer before they could be read).
    /// Skipped samples will be presented as [`MISSED_SAMPLE`] in a `samples` buffer, but the tail of
    /// the ring buffer will be copied to `samples` anyway and the size of the resulting vector will be consistent
    /// with a total amount of samples so far.
    pub fn drain_samples(&self, samples: &mut Vec<u64>) -> std::result::Result<(), usize> {
        let n = self.sample_count();
        let n_drained = samples.len();
        assert!(
            n >= n_drained,
            "Sample buffer is greater that amount of generated samples so far"
        );

        let (copy_range, skipped) = if n > n_drained + NUM_SLOTS {
            // We were not able to keep up. Fill the missing gap with max value.
            // SAFETY: No overflow. n - NUM_SLOTS > n_drained, hence n - NUM_SLOTS > 0
            samples.resize(n - NUM_SLOTS, MISSED_SAMPLE);
            (n - NUM_SLOTS..n, (n - n_drained) - NUM_SLOTS)
        } else {
            (n_drained..n, 0)
        };

        // Draining samples
        for i in copy_range {
            let slot = i & (NUM_SLOTS - 1);
            samples.push(self.samples[slot].load(Ordering::Acquire));
        }
        // number of samples updated during drain. We need to mark those as missed as well
        let inf_flight_samples = self.sample_count().saturating_sub(n);
        if inf_flight_samples > 0 {
            samples[n_drained..(n_drained + inf_flight_samples)].fill(MISSED_SAMPLE);
        }
        if skipped > 0 {
            Err(skipped)
        } else {
            Ok(())
        }
    }

    fn reset(&mut self) {
        self.write_cursor.store(0, Ordering::Release);
        for s in &mut self.samples {
            s.store(0, Ordering::Relaxed);
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

/// Handle to a commpage backed by shared memory.
pub struct Commpage {
    shmem: Shmem,
}

impl Commpage {
    /// Create a new commpage with a unique OS name. Initializes header and both lanes.
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

    /// Get lane C (candidate lane).
    pub fn lane_c(&self) -> &Lane {
        &self.layout().lane_c
    }

    /// Get lane B (baseline lane).
    pub fn lane_b(&self) -> &Lane {
        &self.layout().lane_b
    }

    /// Either candidate or baseline lane has DONE bit set
    pub fn is_some_lane_done(&self) -> bool {
        self.lane_c().is_done() || self.lane_b().is_done()
    }

    /// Get the lane a given role writes to.
    pub fn get_lane(&self, role: Role) -> &Lane {
        match role {
            Role::Candidate => self.lane_c(),
            Role::Baseline => self.lane_b(),
        }
    }

    /// Get the peer's lane (the one a given role reads for synchronization).
    pub fn peer_lane(&self, role: Role) -> &Lane {
        match role {
            Role::Candidate => self.lane_b(),
            Role::Baseline => self.lane_c(),
        }
    }

    /// Set the STOP flag. Called by runner to signal time-budget exhaustion.
    pub fn set_stop(&self) {
        self.layout().flags.fetch_or(STOP_FLAG, Ordering::Release);
    }

    /// Check if the STOP flag is set.
    pub fn is_stopped(&self) -> bool {
        self.layout().flags.load(Ordering::Acquire) & STOP_FLAG != 0
    }

    /// Reset the commpage for a new benchmark run.
    /// Clears flags and resets both lanes' write_cursors to 0.
    pub fn reset(&mut self) {
        self.layout().flags.store(0, Ordering::Release);
        self.layout_mut().lane_c.reset();
        self.layout_mut().lane_b.reset();
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
    fn num_slots_is_a_power_of_2() {
        assert_eq!(
            NUM_SLOTS.count_ones(),
            1,
            "NUM_SLOTS should be power of two"
        );
    }

    #[test]
    fn push_and_drain() {
        let cp = Commpage::create().unwrap();
        let lane = cp.lane_c();

        lane.push_sample(0, 100);
        lane.push_sample(1, 200);
        lane.push_sample(2, 300);

        let mut samples = Vec::new();
        let result = lane.drain_samples(&mut samples);
        assert_eq!(result, Ok(()));
        assert_eq!(samples, vec![100, 200, 300]);

        let result = lane.drain_samples(&mut samples);
        assert_eq!(result, Ok(()));
        assert_eq!(samples, vec![100, 200, 300]);
    }

    #[test]
    fn stop_flag() {
        let cp = Commpage::create().unwrap();
        assert!(!cp.is_stopped());
        cp.set_stop();
        assert!(cp.is_stopped());
    }

    #[test]
    fn done_bit() {
        let cp = Commpage::create().unwrap();
        let lane = cp.lane_c();
        assert!(!lane.is_done());
        lane.push_sample(0, 42);
        lane.mark_done();

        assert!(lane.is_done());
        assert_eq!(lane.sample_count(), 1);
    }

    #[test]
    fn peer_synchronization_two_threads() {
        let cp = Commpage::create().unwrap();
        let id = cp.os_id().to_string();

        let num_samples = 64u64;

        let handle = thread::spawn(move || {
            let cp = Commpage::open(&id).unwrap();
            let my_lane = cp.lane_c();
            let peer_lane = cp.lane_b();

            for i in 0..num_samples {
                my_lane.push_sample(i, i * 10);
                peer_lane.wait_for_cursor(i + 1);
            }
            my_lane.mark_done();
        });

        let my_lane = cp.lane_b();
        let peer_lane = cp.lane_c();

        for i in 0..num_samples {
            my_lane.push_sample(i, i * 20);
            peer_lane.wait_for_cursor(i + 1);
        }
        my_lane.mark_done();

        handle.join().unwrap();

        let mut c_samples = Vec::new();
        let mut b_samples = Vec::new();
        // Verify samples
        cp.lane_c().drain_samples(&mut c_samples).unwrap();
        cp.lane_b().drain_samples(&mut b_samples).unwrap();
        assert_eq!(c_samples.len(), num_samples as usize);
        assert_eq!(b_samples.len(), num_samples as usize);
        for i in 0..num_samples as usize {
            assert_eq!(c_samples[i], i as u64 * 10);
            assert_eq!(b_samples[i], i as u64 * 20);
        }
    }

    #[test]
    fn drain_detects_skipped_samples() {
        let cp = Commpage::create().unwrap();
        let lane = cp.lane_c();

        // Write more than NUM_SLOTS samples
        for i in 0..NUM_SLOTS as u64 + 10 {
            lane.push_sample(i, i);
        }

        let total = NUM_SLOTS + 10;
        let mut samples = Vec::new();
        let result = lane.drain_samples(&mut samples);
        assert_eq!(samples.len(), total);
        assert_eq!(result, Err(10));

        // 10 samples are missed and should be represented in the vector appropriately
        assert_eq!(samples[0..10], vec![MISSED_SAMPLE; 10]);
        for (i, s) in samples[10..].iter().enumerate() {
            assert_eq!(*s, (i + 10) as u64);
        }
    }
}
