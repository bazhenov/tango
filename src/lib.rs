use statrs::distribution::Normal;
use std::{
    collections::BTreeMap,
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use timer::{ActiveTimer, Timer};

pub mod cli;

pub fn benchmark_fn<P, O, F: Fn(&P) -> O>(
    name: impl Into<String>,
    func: F,
) -> impl BenchmarkFn<P, O> {
    let name = name.into();
    assert!(!name.is_empty());
    Func { name, func }
}

pub fn benchmark_fn_with_setup<P, O, I, F: Fn(I) -> O, S: Fn(&P) -> I>(
    name: impl Into<String>,
    func: F,
    setup: S,
) -> impl BenchmarkFn<P, O> {
    let name = name.into();
    assert!(!name.is_empty());
    SetupFunc { name, func, setup }
}

pub trait BenchmarkFn<P, O> {
    fn measure(&self, payload: &P) -> u64;
    fn name(&self) -> &str;
}

struct Func<F> {
    func: F,
    name: String,
}

impl<F, P, O> BenchmarkFn<P, O> for Func<F>
where
    F: Fn(&P) -> O,
{
    fn measure(&self, payload: &P) -> u64 {
        let start = ActiveTimer::start();
        let result = black_box((self.func)(payload));
        let time = ActiveTimer::stop(start);
        drop(result);
        time
    }

    fn name(&self) -> &str {
        self.name.as_str()
    }
}

struct SetupFunc<S, F> {
    setup: S,
    func: F,
    name: String,
}

impl<S, F, P, I, O> BenchmarkFn<P, O> for SetupFunc<S, F>
where
    S: Fn(&P) -> I,
    F: Fn(I) -> O,
{
    fn measure(&self, payload: &P) -> u64 {
        let payload = (self.setup)(payload);
        let start = ActiveTimer::start();
        let result = black_box((self.func)(payload));
        let time = ActiveTimer::stop(start);
        drop(result);
        time
    }

    fn name(&self) -> &str {
        self.name.as_str()
    }
}

pub trait Generator {
    type Output;
    fn next_payload(&mut self) -> Self::Output;
}

pub struct StaticValue<T>(pub T);

impl<T: Copy> Generator for StaticValue<T> {
    type Output = T;

    fn next_payload(&mut self) -> Self::Output {
        self.0
    }
}

pub trait Reporter {
    fn on_complete(&mut self, results: &RunResults);
}

#[derive(Copy, Clone)]
pub enum RunMode {
    Iterations(NonZeroUsize),
    Time(Duration),
}

type FnPair<P, O> = (Box<dyn BenchmarkFn<P, O>>, Box<dyn BenchmarkFn<P, O>>);

pub struct Benchmark<P, O> {
    funcs: BTreeMap<String, FnPair<P, O>>,
    reporters: Vec<Box<dyn Reporter>>,
}

impl<P, O> Default for Benchmark<P, O> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RunOpts {
    name_filter: Option<String>,

    /// Directory location for CSV dump of individual mesurements
    ///
    /// The format is as follows
    /// ```txt
    /// b_1,c_1
    /// b_2,c_2
    /// ...
    /// b_n,c_n
    /// ```
    /// where `b_1..b_n` are baseline absolute time (in nanoseconds) measurements
    /// and `c_1..c_n` are candidate time measurements
    measurements_path: Option<PathBuf>,
    run_mode: RunMode,
    outlier_detection_enabled: bool,
}

impl<P, O> Benchmark<P, O> {
    pub fn new() -> Self {
        Self {
            funcs: BTreeMap::new(),
            reporters: vec![],
        }
    }

    pub fn add_reporter(&mut self, reporter: impl Reporter + 'static) {
        self.reporters.push(Box::new(reporter))
    }

    pub fn add_pair(
        &mut self,
        baseline: impl BenchmarkFn<P, O> + 'static,
        candidate: impl BenchmarkFn<P, O> + 'static,
    ) {
        let key = format!("{}-{}", baseline.name(), candidate.name());
        self.funcs
            .insert(key, (Box::new(baseline), Box::new(candidate)));
    }

    pub fn run_by_name(&mut self, payloads: &mut dyn Generator<Output = P>, opts: &RunOpts) {
        let name_filter = opts.name_filter.as_deref().unwrap_or("");

        for (key, (baseline, candidate)) in &self.funcs {
            if key.contains(name_filter) {
                let dump_path = dump_location(key.as_ref(), opts.measurements_path.as_ref());
                let (base_summary, candidate_summary, diff) = Self::measure(
                    payloads,
                    baseline.as_ref(),
                    candidate.as_ref(),
                    opts.run_mode,
                    dump_path,
                );

                let n = diff.len();

                let diff_summary = if opts.outlier_detection_enabled {
                    let (min, max) =
                        outliers_threshold(diff.to_vec()).unwrap_or((i64::MIN, i64::MAX));

                    let measurements = diff
                        .iter()
                        .copied()
                        .filter(|i| min < *i && *i < max)
                        .collect::<Vec<_>>();
                    Summary::from(&measurements).unwrap()
                } else {
                    Summary::from(&diff).unwrap()
                };

                let outliers_filtered = n - diff_summary.n;

                let std_dev = diff_summary.variance.sqrt();
                let std_err = std_dev / (diff_summary.n as f64).sqrt();
                let z_score = diff_summary.mean / std_err;

                let significant = z_score.abs() >= 2.6;

                let results = RunResults {
                    base_name: baseline.name().to_owned(),
                    candidate_name: candidate.name().to_owned(),
                    base: base_summary,
                    candidate: candidate_summary,
                    diff: diff_summary,
                    significant,
                    outliers: outliers_filtered,
                    n: diff_summary.n,
                };

                for reporter in self.reporters.iter_mut() {
                    reporter.on_complete(&results);
                }
            }
        }
    }

    pub fn run_calibration(&mut self) {
        todo!();
    }

    fn measure(
        generator: &mut dyn Generator<Output = P>,
        base: &dyn BenchmarkFn<P, O>,
        candidate: &dyn BenchmarkFn<P, O>,
        run_mode: RunMode,
        dump_path: Option<impl AsRef<Path>>,
    ) -> (Summary<i64>, Summary<i64>, Vec<i64>) {
        let mut base_time = vec![];
        let mut candidate_time = vec![];

        match run_mode {
            RunMode::Iterations(iter) => {
                for i in 0..usize::from(iter) {
                    let payload = generator.next_payload();
                    if i % 2 == 0 {
                        base_time.push(base.measure(&payload) as i64);
                        candidate_time.push(candidate.measure(&payload) as i64);
                    } else {
                        candidate_time.push(candidate.measure(&payload) as i64);
                        base_time.push(base.measure(&payload) as i64);
                    }
                }
            }
            RunMode::Time(duration) => {
                let deadlline = Instant::now() + duration;
                let mut baseline_first = false;
                while Instant::now() < deadlline {
                    let payload = generator.next_payload();
                    baseline_first = !baseline_first;
                    if baseline_first {
                        base_time.push(base.measure(&payload) as i64);
                        candidate_time.push(candidate.measure(&payload) as i64);
                    } else {
                        candidate_time.push(candidate.measure(&payload) as i64);
                        base_time.push(base.measure(&payload) as i64);
                    }
                }
            }
        }

        if let Some(path) = dump_path {
            write_raw_measurements(path, &base_time, &candidate_time);
        }

        let base = Summary::from(&base_time).unwrap();
        let candidate = Summary::from(&candidate_time).unwrap();
        let diff = base_time
            .into_iter()
            .zip(candidate_time)
            .map(|(b, c)| c - b)
            .collect();
        (base, candidate, diff)
    }

    pub fn list_functions(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(String::as_str)
    }
}

/// Describes the results of a single benchmark run
pub struct RunResults {
    /// name of a baseline function
    pub base_name: String,

    /// name of a candidate function
    pub candidate_name: String,

    /// statistical summary of baseline function measurements
    pub base: Summary<i64>,

    /// statistical summary of candidate function measurements
    pub candidate: Summary<i64>,

    /// individual measurements of a benchmark (candidate - baseline)
    pub diff: Summary<i64>,

    /// Is difference is statistically significant
    pub significant: bool,

    /// Numbers of observations (after outliers filtering)
    pub n: usize,

    /// Numbers of detected and filtered outliers
    pub outliers: usize,
}

fn write_raw_measurements(path: impl AsRef<Path>, base: &[i64], candidate: &[i64]) {
    let mut file = BufWriter::new(File::create(path).unwrap());

    for (b, c) in base.iter().zip(candidate) {
        writeln!(&mut file, "{},{}", b, c).unwrap();
    }
}

fn dump_location(name: &str, dir: Option<impl AsRef<Path>>) -> Option<impl AsRef<Path>> {
    dir.map(|p| p.as_ref().join(format!("{}.csv", name)))
}

/// Statistical summary for a given iterator of numbers.
///
/// Calculates all the information using single pass over the data. Mean and variance are calculated using
/// streaming algorithm described in [1].
///
/// [1]: Art of Computer Programming, Vol 2, page 232
#[derive(Clone, Copy)]
pub struct Summary<T> {
    n: usize,
    min: T,
    max: T,
    mean: f64,
    variance: f64,
}

impl<'a, T: Ord + Copy + 'a> Summary<T> {
    pub fn from<C>(values: C) -> Option<Self>
    where
        i64: From<T>,
        C: IntoIterator<Item = &'a T>,
    {
        Self::running(values.into_iter().copied())?.last()
    }

    pub fn running<I>(mut iter: I) -> Option<impl Iterator<Item = Summary<T>>>
    where
        i64: From<T>,
        I: Iterator<Item = T>,
    {
        let head = iter.next()?;
        Some(RunningSummary {
            iter,
            n: 1,
            min: head,
            max: head,
            mean: i64::from(head) as f64,
            s: 0.,
        })
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
    T: Copy + Ord,
    I: Iterator<Item = T>,
    i64: From<T>,
{
    type Item = Summary<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.iter.next()?;
        let fvalue = i64::from(value) as f64;

        self.min = self.min.min(value);
        self.max = self.max.max(value);

        self.n += 1;
        let mean_p = self.mean;
        self.mean += (fvalue - self.mean) / self.n as f64;
        self.s += (fvalue - mean_p) * (fvalue - self.mean);

        Some(Summary {
            n: self.n,
            min: self.min,
            max: self.max,
            mean: self.mean,
            variance: self.s / (self.n - 1) as f64,
        })
    }
}

/// Running variance iterator
///
/// Provides a running (streaming variance) for a given iterator of observations.
/// Uses simple variance formula: `Var(X) = E[X^2] - E[X]^2`.
pub struct RunningVariance<T> {
    iter: T,
    m: f64,
    s: f64,
    n: f64,
}

impl<T: Iterator<Item = i64>> Iterator for RunningVariance<T> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.iter.next()? as f64;
        self.n += 1.;
        let m_p = self.m;
        self.m += (value - self.m) / self.n;
        self.s += (value - m_p) * (value - self.m);

        if self.n == 1. {
            Some(0.)
        } else {
            Some(self.s / (self.n - 1.))
        }
    }
}

impl<I, T: Iterator<Item = I>> From<T> for RunningVariance<T> {
    fn from(value: T) -> Self {
        Self {
            iter: value,
            m: 0.,
            s: 0.,
            n: 0.,
        }
    }
}

/// Outlier threshold detection
///
/// This functions detects optimal threshold for outlier filtering. Algorithm finds a threshold
/// that split the set of all observations `M` into two different subsets `S` and `O`. Each observation
/// is considered as a split point. Algorithm chooses split point in such way that it maximizes
/// the ration of `S` with this observation and without.
///
/// For example in a set of observations `[1, 2, 3, 100, 200, 300]` the target observation will be 100.
/// It is the observation including which will raise variance the most.
fn outliers_threshold(mut input: Vec<i64>) -> Option<(i64, i64)> {
    // TODO(bazhenov) sorting should be done by difference with median
    input.sort_by_key(|a| a.abs());
    let variance = RunningVariance::from(input.iter().copied());

    // Looking only 30% topmost values
    let mut outliers_cnt = input.len() * 30 / 100;
    let skip = input.len() - outliers_cnt;
    let mut candidate_outliers = input[skip..].iter().filter(|i| **i < 0).count();
    let value_and_variance = input.iter().copied().zip(variance).skip(skip);

    let mut prev_variance = 0.;
    for (value, var) in value_and_variance {
        if prev_variance > 0. && var / prev_variance > 1.2 {
            if let Some((min, max)) = binomial_interval_approximation(outliers_cnt, 0.5) {
                if candidate_outliers < min || candidate_outliers > max {
                    continue;
                }
            }
            return Some((-value.abs(), value.abs()));
        }
        prev_variance = var;
        outliers_cnt -= 1;
        if value < 0 {
            candidate_outliers -= 1;
        }
    }

    None
}

fn binomial_interval_approximation(n: usize, p: f64) -> Option<(usize, usize)> {
    use statrs::distribution::ContinuousCDF;
    let nf = n as f64;
    if nf * p < 10. || nf * (1. - p) < 10. {
        return None;
    }
    let mu = nf * p;
    let sigma = (nf * p * (1. - p)).sqrt();
    let distribution = Normal::new(mu, sigma).unwrap();
    let min = distribution.inverse_cdf(0.1).floor() as usize;
    let max = n - min;
    Some((min, max))
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, RngCore, SeedableRng};
    use std::iter::Sum;

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
        let variances = RunningVariance::from(input.into_iter()).collect::<Vec<_>>();
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
        let mut variances = RunningVariance::from(rng);

        assert!(variances.nth(10000000).unwrap() > 0.)
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

    #[test]
    fn check_filter_outliers() {
        let input = vec![
            1i64, -2, 3, -4, 5, -6, 7, -8, 9, -10, //
            101, -102,
        ];

        let (min, max) = outliers_threshold(input).unwrap();
        assert!(min < 1, "Minimum is: {}", min);
        assert!(10 < max && max <= 101, "Maximum is: {}", max);
    }

    #[test]
    fn check_binomial_approximation() {
        assert_eq!(
            binomial_interval_approximation(10000000, 0.5),
            Some((4997973, 5002027))
        );
    }
}
