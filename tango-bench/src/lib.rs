use num_traits::{AsPrimitive, ToPrimitive};
use std::{
    any::type_name,
    cell::RefCell,
    cmp::Ordering,
    hint::black_box,
    io,
    ops::{Add, Div, RangeInclusive},
    rc::Rc,
    str::Utf8Error,
    time::Duration,
};
use thiserror::Error;
use timer::{ActiveTimer, Timer};

pub mod cli;
pub mod dylib;
#[cfg(target_os = "linux")]
pub mod linux;

pub const NS_TO_MS: u64 = 1_000_000;

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

    #[error("IO Error")]
    IOError(#[from] io::Error),
}

/// Registers benchmark in the system
#[macro_export]
macro_rules! benchmarks {
    ($($func_name:expr),+) => {
        #[no_mangle]
        pub fn __tango_create_benchmarks() -> Vec<Box<dyn tango_bench::MeasureTarget>> {
            use tango_bench::IntoBenchmarks;

            let mut benchmarks = vec![];
            $(benchmarks.extend($func_name.into_benchmarks());)*
            benchmarks
        }
    };
}

pub fn benchmark_fn<O, F: Fn() -> O + 'static>(
    name: &'static str,
    func: F,
) -> Box<dyn MeasureTarget> {
    assert!(!name.is_empty());
    Box::new(SimpleFunc { name, func })
}

pub trait MeasureTarget: Named {
    /// Measures the performance if the function
    ///
    /// Returns the cumulative (all iterations) execution time with nanoseconds precision,
    /// but not necessarily accuracy.
    ///
    /// This method should use the same arguments for measuring the test function unless [`next_haystack()`]
    /// method is called. Only then new set of input arguments should be generated. Although it is allowed
    /// to call this method without first calling [`next_haystack()`]. In which case first haystack should be
    /// generated automatically.
    fn measure(&mut self, iterations: usize) -> u64;

    /// Estimates the number of iterations achievable within given number of miliseconds
    ///
    /// Estimate can be an approximation. If possible the same input arguments should be used when building the
    /// estimate. If the single call to measured function is longer than provided timespan the implementation
    /// can return 0.
    fn estimate_iterations(&mut self, time_ms: u32) -> usize;

    /// Generates next haystack for the measurement
    ///
    /// Calling this method should update internal haystack used for measurement. Returns `true` if update happend,
    /// `false` if implementation doesn't support haystack generation.
    fn next_haystack(&mut self) -> bool;
}

pub trait Named {
    /// The name of the test function
    fn name(&self) -> &str;
}

struct SimpleFunc<F> {
    name: &'static str,
    func: F,
}

impl<O, F: Fn() -> O> MeasureTarget for SimpleFunc<F> {
    fn measure(&mut self, iterations: usize) -> u64 {
        let mut result = Vec::with_capacity(iterations);
        let start = ActiveTimer::start();
        for _ in 0..iterations {
            result.push(black_box((self.func)()));
        }
        let time = ActiveTimer::stop(start);
        drop(result);
        time
    }

    fn estimate_iterations(&mut self, time_ms: u32) -> usize {
        let median = median_execution_time(self, 10) as usize;
        time_ms as usize * 1_000_000 / median
    }

    fn next_haystack(&mut self) -> bool {
        false
    }
}

impl<F> Named for SimpleFunc<F> {
    fn name(&self) -> &str {
        self.name
    }
}

pub struct GenFunc<F, G, H> {
    f: Rc<RefCell<F>>,
    g: Rc<RefCell<G>>,
    haystack: Option<H>,
    name: String,
}

impl<F, H, N, O, G> GenFunc<F, G, H>
where
    G: Generator<Haystack = H, Needle = N>,
    F: Fn(&H, &N) -> O,
{
    pub fn new(name: &str, f: Rc<RefCell<F>>, g: Rc<RefCell<G>>) -> Self {
        let name = format!("{}/{}", name, g.borrow().name());
        Self {
            f,
            g,
            name,
            haystack: None,
        }
    }
}

impl<F, H, N, O, G> MeasureTarget for GenFunc<F, G, H>
where
    G: Generator<Haystack = H, Needle = N>,
    F: Fn(&H, &N) -> O,
{
    fn measure(&mut self, iterations: usize) -> u64 {
        let mut g = self.g.borrow_mut();
        let haystack = &*self.haystack.get_or_insert_with(|| g.next_haystack());

        let f = self.f.borrow_mut();
        let mut result = Vec::with_capacity(iterations);
        let start = ActiveTimer::start();
        for _ in 0..iterations {
            let needle = g.next_needle(&haystack);
            result.push(black_box((f)(haystack, &needle)));
        }
        let time = ActiveTimer::stop(start);
        drop(result);
        time
    }

    fn estimate_iterations(&mut self, time_ms: u32) -> usize {
        // Here we relying on the fact that measure() is not generating a new haystack
        // without a call to next_haystack()
        let measurements = (0..10).map(|_| self.measure(1)).collect::<Vec<_>>();
        (time_ms as usize * 1_000_000) / median(measurements) as usize
    }

    fn next_haystack(&mut self) -> bool {
        self.haystack = Some(self.g.borrow_mut().next_haystack());
        true
    }
}

impl<F, H, N> Named for GenFunc<F, H, N> {
    fn name(&self) -> &str {
        &self.name
    }
}

pub struct BenchmarkMatrix<G> {
    generators: Vec<Rc<RefCell<G>>>,
    functions: Vec<Box<dyn MeasureTarget>>,
}

impl<H: 'static, N, G: Generator<Haystack = H, Needle = N> + 'static> BenchmarkMatrix<G> {
    pub fn new(generator: G) -> Self {
        let generator = Rc::new(RefCell::new(generator));
        Self {
            generators: vec![generator],
            functions: vec![],
        }
    }

    pub fn with_params<P>(params: impl IntoIterator<Item = P>, generator: impl Fn(P) -> G) -> Self {
        let generators: Vec<_> = params
            .into_iter()
            .map(generator)
            .map(RefCell::new)
            .map(Rc::new)
            .collect();
        Self {
            generators,
            functions: vec![],
        }
    }

    pub fn add_function<O, F>(mut self, name: &str, f: F) -> Self
    where
        F: Fn(&H, &N) -> O + 'static,
    {
        let f = Rc::new(RefCell::new(f));
        self.generators
            .iter()
            .map(Rc::clone)
            .map(|g| GenFunc::new(name, Rc::clone(&f), g))
            .map(Box::new)
            .for_each(|f| self.functions.push(f));
        self
    }
}

impl<G> IntoBenchmarks for BenchmarkMatrix<G> {
    fn into_benchmarks(self) -> Vec<Box<dyn MeasureTarget>> {
        self.functions
    }
}

pub trait IntoBenchmarks {
    fn into_benchmarks(self) -> Vec<Box<dyn MeasureTarget>>;
}

impl<const N: usize> IntoBenchmarks for [Box<dyn MeasureTarget>; N] {
    fn into_benchmarks(self) -> Vec<Box<dyn MeasureTarget>> {
        self.into_iter().collect()
    }
}

impl IntoBenchmarks for Vec<Box<dyn MeasureTarget>> {
    fn into_benchmarks(self) -> Vec<Box<dyn MeasureTarget>> {
        self
    }
}

/// Generates the payload for the benchmarking functions
///
/// Generator provides two type of values to the tested functions: *haystack* and *needle*.
///
/// ## Haystack
/// Haystack is typically some sort of a collection that is used in benchmarking.
///
/// ## Needle
/// Needle is some type of query that is presented to the algorithm. In case of searching algorithm, usually it is the
/// value we search in the collection.
///
/// It might be the case that algorithm being tested is not using both type of values. In this case corresponding value
/// type should unit type  –`()`.
pub trait Generator {
    type Haystack;
    type Needle;

    /// Generates next random haystack for the benchmark
    ///
    /// The number of generated haystacks is controlled by [`MeasurementSettings::samples_per_haystack`]
    fn next_haystack(&mut self) -> Self::Haystack;

    fn next_needles(
        &mut self,
        haystack: &Self::Haystack,
        size: usize,
        needles: &mut Vec<Self::Needle>,
    ) {
        for _ in 0..size {
            needles.push(self.next_needle(haystack));
        }
    }

    /// Generates next random needle for the benchmark
    fn next_needle(&mut self, haystack: &Self::Haystack) -> Self::Needle;

    fn name(&self) -> &str {
        let name = type_name::<Self>();
        if let Some(idx) = name.rfind("::") {
            // it's safe to operate on byte offsets here because ':' symbols is 1-byte ascii
            &name[idx + 2..]
        } else {
            name
        }
    }

    fn reset(&mut self) {}
}

pub trait Reporter {
    fn on_complete(&mut self, _results: &RunResult) {}
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
///     max_samples: 10_000,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MeasurementSettings {
    pub max_samples: usize,
    pub max_duration: Duration,
    pub outlier_detection_enabled: bool,

    /// The number of samples per one generated haystack
    pub samples_per_haystack: usize,

    /// Minimum number of iterations in a sample for each of 2 tested functions
    pub min_iterations_per_sample: usize,

    /// The number of iterations in a sample for each of 2 tested functions
    pub max_iterations_per_sample: usize,
}

impl Default for MeasurementSettings {
    fn default() -> Self {
        Self {
            max_samples: 1_000_000,
            max_duration: Duration::from_millis(100),
            outlier_detection_enabled: true,
            samples_per_haystack: 1,
            min_iterations_per_sample: 1,
            max_iterations_per_sample: 50,
        }
    }
}

pub fn calculate_run_result<N: Into<String>>(
    name: N,
    mut baseline: Vec<u64>,
    mut candidate: Vec<u64>,
    iterations_per_sample: usize,
    filter_outliers: bool,
) -> Result<RunResult, Error> {
    assert!(baseline.len() == candidate.len());

    let mut diff = candidate
        .iter()
        .copied()
        .zip(baseline.iter().copied())
        // need to convert both of measurement to i64 because difference can be negative
        .map(|(c, b)| (c as i64, b as i64))
        .map(|(c, b)| (c - b) / iterations_per_sample as i64)
        .collect::<Vec<i64>>();

    let n = diff.len();

    // Normalizing measurements
    for v in baseline.iter_mut() {
        *v /= iterations_per_sample as u64;
    }
    for v in candidate.iter_mut() {
        *v /= iterations_per_sample as u64;
    }

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
                baseline.swap_remove(i);
                candidate.swap_remove(i);
            }
        }
    };

    let diff = Summary::from(&diff).ok_or(Error::NoMeasurements)?;
    let baseline = Summary::from(&baseline).ok_or(Error::NoMeasurements)?;
    let candidate = Summary::from(&candidate).ok_or(Error::NoMeasurements)?;

    let std_dev = diff.variance.sqrt();
    let std_err = std_dev / (diff.n as f64).sqrt();
    let z_score = diff.mean / std_err;

    Ok(RunResult {
        baseline,
        candidate,
        diff,
        name: name.into(),
        // significant result is far away from 0 and have more than 0.5%
        // base/candidate difference
        // z_score = 2.6 corresponds to 99% significance level
        significant: z_score.abs() >= 2.6 && (diff.mean / candidate.mean).abs() > 0.005,
        outliers: n - diff.n,
    })
}

/// Describes the results of a single benchmark run
pub struct RunResult {
    /// name of a test
    pub name: String,

    /// statistical summary of baseline function measurements
    pub baseline: Summary<u64>,

    /// statistical summary of candidate function measurements
    pub candidate: Summary<u64>,

    /// individual measurements of a benchmark (candidate - baseline)
    pub diff: Summary<i64>,

    /// Is difference is statistically significant
    pub significant: bool,

    /// Numbers of detected and filtered outliers
    pub outliers: usize,
}

/// Statistical summary for a given iterator of numbers.
///
/// Calculates all the information using single pass over the data. Mean and variance are calculated using
/// streaming algorithm described in [1].
///
/// [1]: Art of Computer Programming, Vol 2, page 232
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
/// Outliers are observations are 5 IQR away from the corresponding quartile.
fn iqr_variance_thresholds(mut input: Vec<i64>) -> Option<RangeInclusive<i64>> {
    const FACTOR: i64 = 5;

    input.sort();
    let (q1, q3) = (input.len() / 4, input.len() * 3 / 4);
    if q1 >= q3 || q3 >= input.len() || input[q1] >= input[q3] {
        return None;
    }
    let iqr = input[q3] - input[q1];

    let low_threshold = input[q1] - iqr * FACTOR;
    let high_threshold = input[q3] + iqr * FACTOR;

    // Calculating the indicies of the thresholds in an dataset
    let low_threshold_idx = match input[0..q1].binary_search(&low_threshold) {
        Ok(idx) => idx,
        Err(idx) => idx,
    };

    let high_threshold_idx = match input[q3..].binary_search(&high_threshold) {
        Ok(idx) => idx,
        Err(idx) => idx,
    };

    if low_threshold_idx == 0 || high_threshold_idx >= input.len() {
        return None;
    }

    // Calculating the equal number of observations which should be removed from each "side" of observations
    let outliers_cnt = low_threshold_idx.min(input.len() - high_threshold_idx);

    Some(input[outliers_cnt]..=(input[input.len() - outliers_cnt]))
}

mod timer {
    use std::time::Instant;

    #[cfg(all(feature = "hw_timer", target_arch = "x86_64"))]
    pub(super) type ActiveTimer = x86::RdtscpTimer;

    #[cfg(not(feature = "hw_timer"))]
    pub(super) type ActiveTimer = PlatformTimer;

    pub(super) trait Timer<T> {
        fn start() -> T;
        fn stop(start_time: T) -> u64;
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

    #[cfg(all(feature = "hw_timer", target_arch = "x86_64"))]
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

fn median_execution_time(target: &mut dyn MeasureTarget, iterations: u32) -> u64 {
    assert!(iterations >= 1);
    let measures: Vec<_> = (0..iterations).map(|_| target.measure(1)).collect();
    median(measures)
}

fn median<T: Copy + Ord + Add<Output = T> + Div<Output = T> + 'static>(mut measures: Vec<T>) -> T
where
    u32: AsPrimitive<T>,
{
    measures.sort();

    let n = measures.len();
    if n % 2 == 0 {
        (measures[n / 2 - 1] + measures[n / 2]) / 2.as_()
    } else {
        measures[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, RngCore, SeedableRng};
    use std::{iter::Sum, thread};

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
        let _ = Summary::from(<&[i64]>::default());
        let _ = Summary::from(<&[u32]>::default());
        let _ = Summary::from(&Vec::<i64>::default());
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
        let mut target = benchmark_fn("foo", move || {
            thread::sleep(Duration::from_millis(expected_delay))
        });

        let median = median_execution_time(target.as_mut(), 10) / NS_TO_MS;
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
}
