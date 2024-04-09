use core::ptr;
use num_traits::ToPrimitive;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{
    cmp::Ordering, hint::black_box, io, mem, ops::RangeInclusive, str::Utf8Error, time::Duration,
};
use thiserror::Error;
use timer::{ActiveTimer, Timer};

pub mod cli;
pub mod dylib;
#[cfg(target_os = "linux")]
pub mod linux;

#[derive(Debug, Error)]
pub enum Error {
    #[error("No measurements given")]
    NoMeasurements,

    #[error("Invalid string pointer from FFI")]
    InvalidFFIString(Utf8Error),

    #[error("Spi::self() was already called")]
    SpiSelfWasMoved,

    #[error("Unable to load library symbol")]
    UnableToLoadSymbol(#[source] libloading::Error),

    #[error("Unknown sampler type. Available options are: flat and linear")]
    UnknownSamplerType,

    #[error("Invalid test name given")]
    InvalidTestName,

    #[error("IO Error")]
    IOError(#[from] io::Error),
}

/// Registers benchmark in the system
///
/// Macros accepts a list of functions that produce any [`IntoBenchmarks`] type. All of the benchmarks
/// created by those functions are registered in the harness.
///
/// ## Example
/// ```rust
/// use std::time::Instant;
/// use tango_bench::{benchmark_fn, IntoBenchmarks, tango_benchmarks};
///
/// fn time_benchmarks() -> impl IntoBenchmarks {
///     [benchmark_fn("current_time", |b| b.iter(|| Instant::now()))]
/// }
///
/// tango_benchmarks!(time_benchmarks());
/// ```
#[macro_export]
macro_rules! tango_benchmarks {
    ($($func_expr:expr),+) => {
        /// Type checking tango_init() function
        const TANGO_INIT: $crate::dylib::ffi::InitFn = tango_init;

        /// Exported function for initializing the benchmark harness
        #[no_mangle]
        unsafe extern "C" fn tango_init() {
            let mut benchmarks = vec![];
            $(benchmarks.extend($crate::IntoBenchmarks::into_benchmarks($func_expr));)*
            $crate::dylib::__tango_init(benchmarks)
        }

    };
}

/// Main entrypoint for benchmarks
///
/// This macro generate `main()` function for the benchmark harness. Can be used in a form with providing
/// measurement settings:
/// ```rust
/// use tango_bench::{tango_main, tango_benchmarks, MeasurementSettings};
///
/// // Register benchmarks
/// tango_benchmarks!([]);
///
/// tango_main!(MeasurementSettings {
///     samples_per_haystack: 1000,
///     min_iterations_per_sample: 10,
///     max_iterations_per_sample: 10_000,
///     ..Default::default()
/// });
/// ```
#[macro_export]
macro_rules! tango_main {
    ($settings:expr) => {
        fn main() -> $crate::cli::Result<std::process::ExitCode> {
            // Initialize Tango for SelfVTable usage
            unsafe { tango_init() };
            $crate::cli::run($settings)
        }
    };
    () => {
        tango_main! {$crate::MeasurementSettings::default()}
    };
}

pub struct Bencher {
    pub seed: u64,
}

impl Bencher {
    pub fn iter<O: 'static, F: FnMut() -> O + 'static>(self, func: F) -> Box<dyn Sampler> {
        Box::new(SimpleSampler { func })
    }
}

struct SimpleSampler<F> {
    func: F,
}

pub trait Sampler {
    fn measure(&mut self, iterations: usize) -> u64;
}

impl<O, F: FnMut() -> O> Sampler for SimpleSampler<F> {
    fn measure(&mut self, iterations: usize) -> u64 {
        let start = ActiveTimer::start();
        for _ in 0..iterations {
            black_box((self.func)());
        }
        ActiveTimer::stop(start)
    }
}

pub struct Benchmark {
    name: String,
    sampler_factory: Box<dyn SamplerFactory>,
}

pub struct BenchmarkState {
    sampler: Box<dyn Sampler>,
}

pub fn benchmark_fn<F: SamplerFactory + 'static>(
    name: impl Into<String>,
    sampler_factory: F,
) -> Benchmark {
    let name = name.into();
    assert!(!name.is_empty());
    Benchmark {
        name,
        sampler_factory: Box::new(sampler_factory),
    }
}

pub trait BenchmarkIteration<T>: FnMut() -> T {}
impl<T: FnMut() -> O, O> BenchmarkIteration<O> for T {}

pub trait SamplerFactory: FnMut(Bencher) -> Box<dyn Sampler> {
    fn create_sampler(&mut self, params: Bencher) -> Box<dyn Sampler> {
        (self)(params)
    }
}

impl<T: FnMut(Bencher) -> Box<dyn Sampler>> SamplerFactory for T {}

impl Benchmark {
    /// Generates next haystack for the measurement
    ///
    /// Calling this method should update internal haystack used for measurement. Returns `true` if update happend,
    /// `false` if implementation doesn't support haystack generation.
    /// Haystack/Needle distinction is described in [`Generator`] trait.
    pub fn prepare_state(&mut self, seed: u64) -> BenchmarkState {
        let sampler = self.sampler_factory.create_sampler(Bencher { seed });
        BenchmarkState { sampler }
    }

    /// Name of the benchmark
    pub fn name(&self) -> &str {
        self.name.as_str()
    }
}

impl BenchmarkState {
    /// Measures the performance if the function
    ///
    /// Returns the cumulative execution time (all iterations) with nanoseconds precision,
    /// but not necessarily accuracy. Usually this time is get by `clock_gettime()` call or some other
    /// platform-specific call.
    ///
    /// This method should use the same arguments for measuring the test function unless [`prepare_state()`]
    /// method is called. Only then new set of input arguments should be generated. It is NOT allowed
    /// to call this method without first calling [`prepare_state()`].
    ///
    /// [`prepare_state()`]: Self::prepare_state()
    pub fn measure(&mut self, iterations: usize) -> u64 {
        self.sampler.as_mut().measure(iterations)
    }

    /// Estimates the number of iterations achievable within given time.
    ///
    /// Time span is given in milliseconds (`time_ms`). Estimate can be an approximation and it is important
    /// for implementation to be fast (in the order of 10 ms).
    /// If possible the same input arguments should be used when building the estimate.
    /// If the single call of a function is longer than provided timespan the implementation should return 0.
    pub fn estimate_iterations(&mut self, time_ms: u32) -> usize {
        let mut iters = 1;
        let time_ns = Duration::from_millis(time_ms as u64).as_nanos() as u64;

        for _ in 0..5 {
            // Never believe short measurements because they are very unreliable. Pretending that
            // measurement at least took 1us guarantees that we won't end up with an unreasonably large number
            // of iterations
            let time = self.measure(iters).max(1_000);
            let time_per_iteration = (time / iters as u64).max(1);
            let new_iters = (time_ns / time_per_iteration) as usize;

            // Do early stop if new estimate has the same order of magnitude. It is good enough.
            if new_iters < 2 * iters {
                return new_iters;
            }

            iters = new_iters;
        }

        iters
    }
}

/// Converts the implementing type into a vector of [`Benchmark`].
pub trait IntoBenchmarks {
    fn into_benchmarks(self) -> Vec<Benchmark>;
}

impl<const N: usize> IntoBenchmarks for [Benchmark; N] {
    fn into_benchmarks(self) -> Vec<Benchmark> {
        self.into_iter().collect()
    }
}

impl IntoBenchmarks for Vec<Benchmark> {
    fn into_benchmarks(self) -> Vec<Benchmark> {
        self
    }
}

pub(crate) trait Reporter {
    fn on_complete(&mut self, results: &RunResult);
}

/// Describes basic settings for the benchmarking process
///
/// This structure is passed to [`cli::run()`].
///
/// Should be created only with overriding needed properties, like so:
/// ```rust
/// use tango_bench::MeasurementSettings;
///
/// let settings = MeasurementSettings {
///     min_iterations_per_sample: 1000,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MeasurementSettings {
    pub filter_outliers: bool,

    /// The number of samples per one generated haystack
    pub samples_per_haystack: usize,

    /// Minimum number of iterations in a sample for each of 2 tested functions
    pub min_iterations_per_sample: usize,

    /// The number of iterations in a sample for each of 2 tested functions
    pub max_iterations_per_sample: usize,

    pub sampler_type: SampleLengthKind,

    /// If true scheduler performs warmup iterations before measuring function
    pub warmup_enabled: bool,

    /// Size of a CPU cache firewall in KBytes
    ///
    /// If set, the scheduler will perform a dummy data read between samples generation to spoil the CPU cache
    ///
    /// Cache firewall is a way to reduce the impact of the CPU cache on the benchmarking process. It tries
    /// to minimize discrepancies in performance between two algorithms due to the CPU cache state.
    pub cache_firewall: Option<usize>,

    /// If true, scheduler will perform a yield of control back to the OS before taking each sample
    ///
    /// Yielding control to the OS is a way to reduce the impact of OS scheduler on the benchmarking process.
    pub yield_before_sample: bool,

    /// If set, use alloca to allocate a random offset for the stack each sample.
    /// This to reduce memory alignment effects on the benchmarking process.
    ///
    /// May cause UB if the allocation is larger then the thread stack size.
    pub randomize_stack: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub enum SampleLengthKind {
    Flat,
    Linear,
    Random,
}

/// Performs a dummy reads from memory to spoil given amount of CPU cache
///
/// Uses cache aligned data arrays to perform minimum amount of reads possible to spoil the cache
struct CacheFirewall {
    cache_lines: Vec<CacheLine>,
}

impl CacheFirewall {
    fn new(bytes: usize) -> Self {
        let n = bytes / mem::size_of::<CacheLine>();
        let cache_lines = vec![CacheLine::default(); n];
        Self { cache_lines }
    }

    fn issue_read(&self) {
        for line in &self.cache_lines {
            // Because CacheLine is aligned on 64 bytes it is enough to read single element from the array
            // to spoil the whole cache line
            unsafe { ptr::read_volatile(&line.0[0]) };
        }
    }
}

#[repr(C)]
#[repr(align(64))]
#[derive(Default, Clone, Copy)]
struct CacheLine([u16; 32]);

pub const DEFAULT_SETTINGS: MeasurementSettings = MeasurementSettings {
    filter_outliers: false,
    samples_per_haystack: 1,
    min_iterations_per_sample: 1,
    max_iterations_per_sample: 5000,
    sampler_type: SampleLengthKind::Random,
    cache_firewall: None,
    yield_before_sample: false,
    warmup_enabled: true,
    randomize_stack: None,
};

impl Default for MeasurementSettings {
    fn default() -> Self {
        DEFAULT_SETTINGS
    }
}

/// Responsible for determining the number of iterations to run for each sample
///
/// Different sampler strategies can influence the results heavily. For example, if function is dependent heavily
/// on a memory subsystem, then it should be tested with different number of iterations to be representative
/// for different memory access patterns and cache states.
trait SampleLength {
    /// Returns the number of iterations to run for the next sample
    ///
    /// Accepts the number of iteration being run starting from 0 and cummulative time spent by both functions
    fn next_sample_iterations(&mut self, iteration_no: usize, estimate: usize) -> usize;
}

/// Runs the same number of iterations for each sample
///
/// Estimates the number of iterations based on the number of iterations achieved in 10 ms and uses
/// this number as a base for the number of iterations for each sample.
struct FlatSampleLength {
    min: usize,
    max: usize,
}

impl FlatSampleLength {
    fn new(settings: &MeasurementSettings) -> Self {
        FlatSampleLength {
            min: settings.min_iterations_per_sample.max(1),
            max: settings.max_iterations_per_sample,
        }
    }
}

impl SampleLength for FlatSampleLength {
    fn next_sample_iterations(&mut self, _iteration_no: usize, estimate: usize) -> usize {
        estimate.clamp(self.min, self.max)
    }
}

struct LinearSampleLength {
    min: usize,
    max: usize,
}

impl LinearSampleLength {
    fn new(settings: &MeasurementSettings) -> Self {
        Self {
            min: settings.min_iterations_per_sample.max(1),
            max: settings.max_iterations_per_sample,
        }
    }
}

impl SampleLength for LinearSampleLength {
    fn next_sample_iterations(&mut self, iteration_no: usize, estimate: usize) -> usize {
        let estimate = estimate.clamp(self.min, self.max);
        (iteration_no % estimate) + 1
    }
}

/// Sampler that randomly determines the number of iterations to run for each sample
///
/// This sampler uses a random number generator to decide the number of iterations for each sample.
struct RandomSampleLength {
    rng: SmallRng,
    min: usize,
    max: usize,
}

impl RandomSampleLength {
    pub fn new(settings: &MeasurementSettings, seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            min: settings.min_iterations_per_sample.max(1),
            max: settings.max_iterations_per_sample,
        }
    }
}

impl SampleLength for RandomSampleLength {
    fn next_sample_iterations(&mut self, _iteration_no: usize, estimate: usize) -> usize {
        let estimate = estimate.clamp(self.min, self.max);
        self.rng.gen_range(1..=estimate)
    }
}

/// Calculates the result of the benchmarking run
///
/// Return None if no measurements were made
pub(crate) fn calculate_run_result<N: Into<String>>(
    name: N,
    baseline: &[u64],
    candidate: &[u64],
    iterations_per_sample: &[usize],
    filter_outliers: bool,
) -> Option<RunResult> {
    assert!(baseline.len() == candidate.len());
    assert!(baseline.len() == iterations_per_sample.len());

    let mut iterations_per_sample = iterations_per_sample.to_vec();

    let mut diff = candidate
        .iter()
        .zip(baseline.iter())
        // Calculating difference between candidate and baseline
        .map(|(&c, &b)| (c as f64 - b as f64))
        .zip(iterations_per_sample.iter())
        // Normalizing difference to iterations count
        .map(|(diff, &iters)| diff / iters as f64)
        .collect::<Vec<_>>();

    // need to save number of original samples to calculate number of outliers correctly
    let n = diff.len();

    // Normalizing measurements to iterations count
    let mut baseline = baseline
        .iter()
        .zip(iterations_per_sample.iter())
        .map(|(&v, &iters)| (v as f64) / (iters as f64))
        .collect::<Vec<_>>();
    let mut candidate = candidate
        .iter()
        .zip(iterations_per_sample.iter())
        .map(|(&v, &iters)| (v as f64) / (iters as f64))
        .collect::<Vec<_>>();

    // Calculating measurements range. All measurements outside this interval concidered outliers
    let range = if filter_outliers {
        iqr_variance_thresholds(diff.to_vec())
    } else {
        None
    };

    // Cleaning measurements from outliers if needed
    if let Some(range) = range {
        // We filtering outliers to build statistical Summary and the order of elements in arrays
        // doesn't matter, therefore swap_remove() is used. But we need to make sure that all arrays
        // has the same length
        assert_eq!(diff.len(), baseline.len());
        assert_eq!(diff.len(), candidate.len());

        let mut i = 0;
        while i < diff.len() {
            if range.contains(&diff[i]) {
                i += 1;
            } else {
                diff.swap_remove(i);
                iterations_per_sample.swap_remove(i);
                baseline.swap_remove(i);
                candidate.swap_remove(i);
            }
        }
    };

    let diff_summary = Summary::from(&diff)?;
    let baseline_summary = Summary::from(&baseline)?;
    let candidate_summary = Summary::from(&candidate)?;

    let diff_estimate = DiffEstimate::build(&baseline_summary, &diff_summary);

    Some(RunResult {
        baseline: baseline_summary,
        candidate: candidate_summary,
        diff: diff_summary,
        name: name.into(),
        diff_estimate,
        outliers: n - diff_summary.n,
    })
}

/// Contains the estimation of how much faster or slower is candidate function compared to baseline
pub(crate) struct DiffEstimate {
    // Percentage of difference between candidate and baseline
    //
    // Negative value means that candidate is faster than baseline, positive - slower.
    pct: f64,

    // Is the difference statistically significant
    significant: bool,
}

impl DiffEstimate {
    /// Builds [`DiffEstimate`] from flat sampling
    ///
    /// Flat sampling is a sampling where each measurement is normalized by the number of iterations.
    /// This is needed to make measurements comparable between each other. Linear sampling is more
    /// robust to outliers, but it is requiring more iterations.
    ///
    /// It is assumed that baseline and candidate are already normalized by iterations count.
    fn build(baseline: &Summary<f64>, diff: &Summary<f64>) -> Self {
        let std_dev = diff.variance.sqrt();
        let std_err = std_dev / (diff.n as f64).sqrt();
        let z_score = diff.mean / std_err;

        // significant result is far away from 0 and have more than 0.5% base/candidate difference
        // z_score = 2.6 corresponds to 99% significance level
        let significant = z_score.abs() >= 2.6
            && (diff.mean / baseline.mean).abs() > 0.005
            && diff.mean.abs() >= ActiveTimer::precision() as f64;
        let pct = diff.mean / baseline.mean * 100.0;

        Self { pct, significant }
    }
}

/// Describes the results of a single benchmark run
pub(crate) struct RunResult {
    /// name of a test
    name: String,

    /// statistical summary of baseline function measurements
    baseline: Summary<f64>,

    /// statistical summary of candidate function measurements
    candidate: Summary<f64>,

    /// individual measurements of a benchmark (candidate - baseline)
    diff: Summary<f64>,

    diff_estimate: DiffEstimate,

    /// Numbers of detected and filtered outliers
    outliers: usize,
}

/// Statistical summary for a given iterator of numbers.
///
/// Calculates all the information using single pass over the data. Mean and variance are calculated using
/// streaming algorithm described in _Art of Computer Programming, Vol 2, page 232_.
#[derive(Clone, Copy)]
pub struct Summary<T> {
    pub n: usize,
    pub min: T,
    pub max: T,
    pub mean: f64,
    pub variance: f64,
}

impl<T: PartialOrd> Summary<T> {
    pub fn from<'a, C>(values: C) -> Option<Self>
    where
        C: IntoIterator<Item = &'a T>,
        T: ToPrimitive + Copy + Default + 'a,
    {
        Self::running(values.into_iter().copied()).last()
    }

    pub fn running<I>(iter: I) -> impl Iterator<Item = Summary<T>>
    where
        T: ToPrimitive + Copy + Default,
        I: Iterator<Item = T>,
    {
        RunningSummary {
            iter,
            n: 0,
            min: T::default(),
            max: T::default(),
            mean: 0.,
            s: 0.,
        }
    }
}

struct RunningSummary<T, I> {
    iter: I,
    n: usize,
    min: T,
    max: T,
    mean: f64,
    s: f64,
}

impl<T, I> Iterator for RunningSummary<T, I>
where
    T: Copy + PartialOrd,
    I: Iterator<Item = T>,
    T: ToPrimitive,
{
    type Item = Summary<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.iter.next()?;
        let fvalue = value.to_f64().expect("f64 overflow detected");

        if self.n == 0 {
            self.min = value;
            self.max = value;
        }

        if let Some(Ordering::Less) = value.partial_cmp(&self.min) {
            self.min = value;
        }
        if let Some(Ordering::Greater) = value.partial_cmp(&self.max) {
            self.max = value;
        }

        self.n += 1;
        let mean_p = self.mean;
        self.mean += (fvalue - self.mean) / self.n as f64;
        self.s += (fvalue - mean_p) * (fvalue - self.mean);
        let variance = if self.n > 1 {
            self.s / (self.n - 1) as f64
        } else {
            0.
        };

        Some(Summary {
            n: self.n,
            min: self.min,
            max: self.max,
            mean: self.mean,
            variance,
        })
    }
}

/// Outlier detection algorithm based on interquartile range
///
/// Observations that are 1.5 IQR away from the corresponding quartile are consideted as outliers
/// as described in original Tukey's paper.
pub fn iqr_variance_thresholds(mut input: Vec<f64>) -> Option<RangeInclusive<f64>> {
    const MINIMUM_IQR: f64 = 1.;

    input.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let (q1, q3) = (input.len() / 4, input.len() * 3 / 4 - 1);
    if q1 >= q3 || q3 >= input.len() {
        return None;
    }
    // In case q1 and q3 are equal, we need to make sure that IQR is not 0
    // In the future it would be nice to measure system timer precision empirically.
    let iqr = (input[q3] - input[q1]).max(MINIMUM_IQR);

    let low_threshold = input[q1] - iqr * 1.5;
    let high_threshold = input[q3] + iqr * 1.5;

    // Calculating the indicies of the thresholds in an dataset
    let low_threshold_idx =
        match input[0..q1].binary_search_by(|probe| probe.total_cmp(&low_threshold)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

    let high_threshold_idx =
        match input[q3..].binary_search_by(|probe| probe.total_cmp(&high_threshold)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

    if low_threshold_idx == 0 || high_threshold_idx >= input.len() {
        return None;
    }

    // Calculating the equal number of observations which should be removed from each "side" of observations
    let outliers_cnt = low_threshold_idx.min(input.len() - high_threshold_idx);

    Some(input[outliers_cnt]..=(input[input.len() - outliers_cnt - 1]))
}

mod timer {
    use std::time::Instant;

    #[cfg(all(feature = "hw-timer", target_arch = "x86_64"))]
    pub(super) type ActiveTimer = x86::RdtscpTimer;

    #[cfg(not(feature = "hw-timer"))]
    pub(super) type ActiveTimer = PlatformTimer;

    pub(super) trait Timer<T> {
        fn start() -> T;
        fn stop(start_time: T) -> u64;

        /// Timer precision in nanoseconds
        ///
        /// The results less than the precision of a timer are considered not significant
        fn precision() -> u64 {
            1
        }
    }

    pub(super) struct PlatformTimer;

    impl Timer<Instant> for PlatformTimer {
        #[inline]
        fn start() -> Instant {
            Instant::now()
        }

        #[inline]
        fn stop(start_time: Instant) -> u64 {
            start_time.elapsed().as_nanos() as u64
        }
    }

    #[cfg(all(feature = "hw-timer", target_arch = "x86_64"))]
    pub(super) mod x86 {
        use super::Timer;
        use std::arch::x86_64::{__rdtscp, _mm_mfence};

        pub struct RdtscpTimer;

        impl Timer<u64> for RdtscpTimer {
            #[inline]
            fn start() -> u64 {
                unsafe {
                    _mm_mfence();
                    __rdtscp(&mut 0)
                }
            }

            #[inline]
            fn stop(start: u64) -> u64 {
                unsafe {
                    let end = __rdtscp(&mut 0);
                    _mm_mfence();
                    end - start
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, Rng, RngCore, SeedableRng};
    use std::{
        iter::Sum,
        ops::{Add, Div},
        thread,
        time::Duration,
    };

    #[test]
    fn check_iqr_variance_thresholds() {
        let mut rng = SmallRng::from_entropy();

        // Generate 20 random values in range [-50, 50]
        // and add 10 outliers in each of two ranges [-1000, -200] and [200, 1000]
        // This way IQR is no more than 100 and thresholds should be withing [-50, 50] range
        let mut values = vec![];
        values.extend((0..20).map(|_| rng.gen_range(-50.0..=50.)));
        values.extend((0..10).map(|_| rng.gen_range(-1000.0..=-200.0)));
        values.extend((0..10).map(|_| rng.gen_range(200.0..=1000.0)));

        let thresholds = iqr_variance_thresholds(values).unwrap();

        assert!(
            -50. <= *thresholds.start() && *thresholds.end() <= 50.,
            "Invalid range: {:?}",
            thresholds
        );
    }

    /// This tests checks that the algorithm is stable in case of zero difference between 25 and 75 percentiles
    #[test]
    fn check_outliers_zero_iqr() {
        let mut rng = SmallRng::from_entropy();

        let mut values = vec![];
        values.extend(std::iter::repeat(0.).take(20));
        values.extend((0..10).map(|_| rng.gen_range(-1000.0..=-200.0)));
        values.extend((0..10).map(|_| rng.gen_range(200.0..=1000.0)));

        let thresholds = iqr_variance_thresholds(values).unwrap();

        assert!(
            0. <= *thresholds.start() && *thresholds.end() <= 0.,
            "Invalid range: {:?}",
            thresholds
        );
    }

    #[test]
    fn check_summary_statistics() {
        for i in 2u32..100 {
            let range = 1..=i;
            let values = range.collect::<Vec<_>>();
            let stat = Summary::from(&values).unwrap();

            let sum = (i * (i + 1)) as f64 / 2.;
            let expected_mean = sum as f64 / i as f64;
            let expected_variance = naive_variance(values.as_slice());

            assert_eq!(stat.min, 1);
            assert_eq!(stat.n, i as usize);
            assert_eq!(stat.max, i);
            assert!(
                (stat.mean - expected_mean).abs() < 1e-5,
                "Expected close to: {}, given: {}",
                expected_mean,
                stat.mean
            );
            assert!(
                (stat.variance - expected_variance).abs() < 1e-5,
                "Expected close to: {}, given: {}",
                expected_variance,
                stat.variance
            );
        }
    }

    #[test]
    fn check_summary_statistics_types() {
        Summary::from(<&[i64]>::default());
        Summary::from(<&[u32]>::default());
        Summary::from(&Vec::<i64>::default());
    }

    #[test]
    fn check_naive_variance() {
        assert_eq!(naive_variance(&[1, 2, 3]), 1.0);
        assert_eq!(naive_variance(&[1, 2, 3, 4, 5]), 2.5);
    }

    #[test]
    fn check_running_variance() {
        let input = [1i64, 2, 3, 4, 5, 6, 7];
        let variances = Summary::running(input.into_iter())
            .map(|s| s.variance)
            .collect::<Vec<_>>();
        let expected = &[0., 0.5, 1., 1.6666, 2.5, 3.5, 4.6666];

        assert_eq!(variances.len(), expected.len());

        for (value, expected_value) in variances.iter().zip(expected) {
            assert!(
                (value - expected_value).abs() < 1e-3,
                "Expected close to: {}, given: {}",
                expected_value,
                value
            );
        }
    }

    #[test]
    fn check_running_variance_stress_test() {
        let rng = RngIterator(SmallRng::seed_from_u64(0)).map(|i| i as i64);
        let mut variances = Summary::running(rng).map(|s| s.variance);

        assert!(variances.nth(1_000_000).unwrap() > 0.)
    }

    /// Basic check of measurement code
    ///
    /// This test is quite brittle. There is no guarantee the OS scheduler will wake up the thread
    /// soon enough to meet measurement target. We try to mitigate this possibility using several strategies:
    /// 1. repeating test several times and taking median as target measurement.
    /// 2. using more liberal checking condition (allowing 1 order of magnitude error in measurement)
    #[test]
    fn check_measure_time() {
        let expected_delay = 1;
        let mut target = benchmark_fn("foo", move |b| {
            b.iter(move || thread::sleep(Duration::from_millis(expected_delay)))
        });
        target.prepare_state(0);

        let median = median_execution_time(&mut target, 10).as_millis() as u64;
        assert!(median < expected_delay * 10);
    }

    struct RngIterator<T>(T);

    impl<T: RngCore> Iterator for RngIterator<T> {
        type Item = u32;

        fn next(&mut self) -> Option<Self::Item> {
            Some(self.0.next_u32())
        }
    }

    fn naive_variance<T>(values: &[T]) -> f64
    where
        T: Sum + Copy,
        f64: From<T>,
    {
        let n = values.len() as f64;
        let mean = f64::from(values.iter().copied().sum::<T>()) / n;
        let mut sum_of_squares = 0.;
        for value in values.into_iter().copied() {
            sum_of_squares += (f64::from(value) - mean).powi(2);
        }
        sum_of_squares / (n - 1.)
    }

    fn median_execution_time(target: &mut Benchmark, iterations: u32) -> Duration {
        assert!(iterations >= 1);
        let mut state = target.prepare_state(0);
        let measures: Vec<_> = (0..iterations).map(|_| state.measure(1)).collect();
        let time = median(measures).max(1);
        Duration::from_nanos(time)
    }

    fn median<T: Copy + Ord + Add<Output = T> + Div<Output = T>>(mut measures: Vec<T>) -> T {
        assert!(!measures.is_empty(), "Vec is empty");
        measures.sort_unstable();
        measures[measures.len() / 2]
    }
}
