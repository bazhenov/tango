use num_traits::{AsPrimitive, ToPrimitive};
use std::{
    any::type_name,
    cmp::Ordering,
    collections::BTreeMap,
    fs::File,
    hint::black_box,
    io::{BufWriter, Write as _},
    ops::{Add, Div},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use timer::{ActiveTimer, Timer};

pub mod cli;
pub mod dylib;

pub const NS_TO_MS: u64 = 1_000_000;

pub fn benchmark_fn<O, F: Fn() -> O + 'static>(
    name: &'static str,
    func: F,
) -> Box<dyn MeasureTarget> {
    assert!(!name.is_empty());
    Box::new(SimpleFunc { name, func })
}

pub const fn _benchmark_fn<H, N, O, F>(name: &'static str, func: F) -> impl BenchmarkFn<H, N>
where
    F: Fn(&H, &N) -> O,
{
    assert!(!name.is_empty());
    Func { name, func }
}

pub fn benchmark_fn_with_setup<H, N, O, I: Clone, F, S>(
    name: impl Into<String>,
    func: F,
    setup: S,
) -> impl BenchmarkFn<H, N>
where
    I: Clone,
    F: Fn(I, &N) -> O,
    S: Fn(&H) -> I,
{
    let name = name.into();
    assert!(!name.is_empty());
    SetupFunc { name, func, setup }
}

pub trait BenchmarkFn<H, N> {
    fn measure(&self, haystack: &H, needles: &[N]) -> u64;
    fn name(&self) -> &str;
}

struct Func<F> {
    name: &'static str,
    func: F,
}

impl<F, H, N, O> BenchmarkFn<H, N> for Func<F>
where
    F: Fn(&H, &N) -> O,
{
    fn measure(&self, haystack: &H, needles: &[N]) -> u64 {
        let iterations = needles.len();
        let mut result = Vec::with_capacity(iterations);
        let start = ActiveTimer::start();
        for needle in needles {
            result.push(black_box((self.func)(haystack, needle)));
        }
        let time = ActiveTimer::stop(start);
        drop(result);
        time
    }

    fn name(&self) -> &str {
        self.name
    }
}

pub trait MeasureTarget {
    /// Measures the performance if the function
    ///
    /// Returns the cumulative (all iterations) execution time with nanoseconds precision,
    /// but not necessarily accuracy.
    fn measure(&mut self, iterations: usize) -> u64;

    /// Estimates the number of iterations achievable within given number of miliseconds
    ///
    /// Estimate can be an approximation. If possible the same input arguments should be used when building the
    /// estimate. If the single call to measured function is longer than provided timespan the implementation
    /// can return 0.
    fn estimate_iterations(&mut self, time_ms: u32) -> usize;

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

    fn name(&self) -> &str {
        self.name
    }
}

pub struct GenAndFunc<H, N> {
    f: Box<dyn BenchmarkFn<H, N>>,
    g: Box<dyn Generator<Haystack = H, Needle = N>>,
    name: String,
}

impl<H, N> GenAndFunc<H, N> {
    pub fn new(
        f: impl BenchmarkFn<H, N> + 'static,
        g: impl Generator<Haystack = H, Needle = N> + 'static,
    ) -> Self {
        let name = format!("{}/{}", f.name(), g.name());
        Self {
            f: Box::new(f),
            g: Box::new(g),
            name,
        }
    }
}

impl<H: 'static, N: 'static> GenAndFunc<H, N> {
    pub fn new_boxed(
        f: impl BenchmarkFn<H, N> + 'static,
        g: impl Generator<Haystack = H, Needle = N> + 'static,
    ) -> Box<dyn MeasureTarget> {
        Box::new(Self::new(f, g))
    }
}

impl<H, N> MeasureTarget for GenAndFunc<H, N> {
    fn measure(&mut self, iterations: usize) -> u64 {
        let haystack = self.g.next_haystack();
        let mut needles = Vec::with_capacity(iterations);
        self.g.next_needles(&haystack, iterations, &mut needles);
        self.f.measure(&haystack, &needles)
    }

    fn estimate_iterations(&mut self, time_ms: u32) -> usize {
        let iterations = 10;

        let haystack = self.g.next_haystack();
        let mut needles = Vec::with_capacity(iterations);
        self.g.next_needles(&haystack, iterations, &mut needles);

        let mut measurements = Vec::with_capacity(iterations);

        for needle in needles {
            measurements.push(self.f.measure(&haystack, &[needle]));
        }

        (time_ms as usize * 1_000_000) / median(measurements) as usize
    }

    fn name(&self) -> &str {
        self.name.as_str()
    }
}

pub struct GeneratorBenchmarks<G> {
    generators: Vec<G>,
    functions: Vec<Box<dyn MeasureTarget>>,
}

impl<H: 'static, N: 'static, G: Generator<Haystack = H, Needle = N> + 'static>
    GeneratorBenchmarks<G>
{
    pub fn with_generator(generator: G) -> Self {
        Self {
            generators: vec![generator],
            functions: vec![],
        }
    }

    pub fn with_generators<P>(
        params: impl IntoIterator<Item = P>,
        generator: impl Fn(P) -> G,
    ) -> Self {
        let generators: Vec<_> = params.into_iter().map(|p| generator(p)).collect();
        Self {
            generators,
            functions: vec![],
        }
    }

    pub fn add<O, F>(&mut self, name: &'static str, f: F) -> &mut Self
    where
        G: Clone,
        O: 'static,
        F: Fn(&H, &N) -> O + Copy + 'static,
    {
        for g in &self.generators {
            let f = _benchmark_fn(name, f);
            self.functions.push(Box::new(GenAndFunc::new(f, g.clone())));
        }
        self
    }
}

impl<G> IntoBenchmarks for GeneratorBenchmarks<G> {
    fn into_benchmarks(self) -> Vec<Box<dyn MeasureTarget>> {
        self.functions
    }
}

pub trait IntoBenchmarks {
    fn into_benchmarks(self) -> Vec<Box<dyn MeasureTarget>>;
}

impl IntoBenchmarks for Vec<Box<dyn MeasureTarget>> {
    fn into_benchmarks(self) -> Vec<Box<dyn MeasureTarget>> {
        self
    }
}

struct SetupFunc<S, F> {
    name: String,
    setup: S,
    func: F,
}

impl<S, F, H, N, I, O> BenchmarkFn<H, N> for SetupFunc<S, F>
where
    S: Fn(&H) -> I,
    F: Fn(I, &N) -> O,
    I: Clone,
{
    fn measure(&self, haystack: &H, needles: &[N]) -> u64 {
        let iterations = needles.len();
        let mut results = Vec::with_capacity(iterations);
        let haystack = (self.setup)(haystack);
        let start = ActiveTimer::start();
        for needle in needles {
            results.push(black_box((self.func)(haystack.clone(), needle)));
        }
        let time = ActiveTimer::stop(start);
        drop(results);
        time
    }

    fn name(&self) -> &str {
        self.name.as_str()
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

    fn name(&self) -> String {
        type_name::<Self>().to_string()
    }

    fn reset(&mut self) {}
}

/// Generator that provides static value to the benchmark. The value should implement [`Copy`] trait.
pub struct StaticValue<H, N>(
    /// Haystack value
    pub H,
    /// Needle value
    pub N,
);

impl<H: Copy, N: Copy> Generator for StaticValue<H, N> {
    type Haystack = H;
    type Needle = N;

    fn next_haystack(&mut self) -> Self::Haystack {
        self.0
    }

    fn next_needle(&mut self, _: &Self::Haystack) -> Self::Needle {
        self.1
    }

    fn name(&self) -> String {
        "StaticValue".to_string()
    }
}

pub trait Reporter {
    fn on_start(&mut self, _generator_name: &str) {}
    fn on_complete(&mut self, _results: &RunResult) {}
}

type FnPair<H, N> = (Box<dyn BenchmarkFn<H, N>>, Box<dyn BenchmarkFn<H, N>>);

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

pub struct Benchmark<H, N> {
    funcs: BTreeMap<String, FnPair<H, N>>,
    generators: Vec<Box<dyn Generator<Haystack = H, Needle = N>>>,
    reporters: Vec<Box<dyn Reporter>>,
}

impl<H, N> Default for Benchmark<H, N> {
    fn default() -> Self {
        Self {
            funcs: BTreeMap::new(),
            generators: vec![],
            reporters: vec![],
        }
    }
}

impl<H, N> Benchmark<H, N> {
    pub fn add_reporter(&mut self, reporter: impl Reporter + 'static) {
        self.reporters.push(Box::new(reporter))
    }

    pub fn add_generator(&mut self, generator: impl Generator<Haystack = H, Needle = N> + 'static) {
        self.generators.push(Box::new(generator))
    }

    pub fn add_generators<T>(&mut self, generators: impl IntoIterator<Item = T>)
    where
        T: Generator<Haystack = H, Needle = N> + 'static,
    {
        for generator in generators {
            self.add_generator(generator);
        }
    }

    pub fn add_pair(
        &mut self,
        baseline: impl BenchmarkFn<H, N> + 'static,
        candidate: impl BenchmarkFn<H, N> + 'static,
    ) {
        let key = format!("{}-{}", baseline.name(), candidate.name());
        self.funcs
            .insert(key, (Box::new(baseline), Box::new(candidate)));
    }

    pub fn run_by_name(
        &mut self,
        reporter: &mut dyn Reporter,
        name_filter: &str,
        opts: &MeasurementSettings,
        samples_dump: Option<impl AsRef<Path>>,
    ) {
        for generator in self.generators.iter_mut() {
            let generator_name = generator.name();
            let mut start_reported = false;
            for (key, (baseline, candidate)) in &self.funcs {
                if key.contains(name_filter) || generator_name.contains(name_filter) {
                    if !start_reported {
                        reporter.on_start(generator_name.as_str());
                        start_reported = true;
                    }

                    let samples_dump = &samples_dump;
                    let (baseline_summary, candidate_summary, diff) = measure_function_pair(
                        generator.as_mut(),
                        baseline.as_ref(),
                        candidate.as_ref(),
                        opts,
                        samples_dump.as_ref(),
                    );

                    let run_result = calculate_run_result(
                        (baseline.name(), baseline_summary),
                        (candidate.name(), candidate_summary),
                        diff,
                        opts.outlier_detection_enabled,
                    );

                    reporter.on_complete(&run_result);
                }
            }
        }
    }

    pub fn run_calibration(&mut self) {
        const TRIES: usize = 10;

        let generator = self.generators[0].as_mut();
        println!("H0 testing...");
        for (baseline, candidate) in self.funcs.values() {
            for f in [baseline.as_ref(), candidate.as_ref()] {
                let significant = Self::calibrate(generator, f, f, TRIES);
                let successes = TRIES - significant;
                println!("    {:30} ... {}/{}", f.name(), successes, TRIES);
            }
        }

        println!("H1 testing...");
        for (baseline, candidate) in self.funcs.values() {
            let significant =
                Self::calibrate(generator, baseline.as_ref(), candidate.as_ref(), TRIES);
            let name = format!("{} / {}", baseline.name(), candidate.name());
            println!("    {:30} ... {}/{}", name, significant, TRIES);
        }
    }

    /// Runs a given test multiple times and return the the number of times difference is statistically significant
    fn calibrate(
        payloads: &mut (dyn Generator<Haystack = H, Needle = N>),
        a: &dyn BenchmarkFn<H, N>,
        b: &dyn BenchmarkFn<H, N>,
        tries: usize,
    ) -> usize {
        let mut succeed = 0;
        let opts = MeasurementSettings {
            max_samples: 1_000_000,
            max_duration: Duration::from_millis(1000),
            ..Default::default()
        };
        for _ in 0..tries {
            let (a_summary, b_summary, diff) =
                measure_function_pair(payloads, a, b, &opts, Option::<PathBuf>::None);

            let result =
                calculate_run_result((a.name(), a_summary), (b.name(), b_summary), diff, true);
            succeed += usize::from(result.significant);
        }
        succeed
    }

    pub fn list_functions(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(String::as_str)
    }
}

/// Measure the difference in performance of two functions
///
/// Provides a way to save a raw dump of measurements into directory
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
fn measure_function_pair<H, N>(
    generator: &mut dyn Generator<Haystack = H, Needle = N>,
    base: &dyn BenchmarkFn<H, N>,
    candidate: &dyn BenchmarkFn<H, N>,
    opts: &MeasurementSettings,
    samples_dump_location: Option<impl AsRef<Path>>,
) -> (Summary<i64>, Summary<i64>, Vec<i64>) {
    let mut base_samples = Vec::with_capacity(opts.max_samples);
    let mut candidate_samples = Vec::with_capacity(opts.max_samples);

    let iterations_per_ms = estimate_iterations_per_ms(generator, base, candidate);
    let deadline = Instant::now() + opts.max_duration;

    generator.reset();
    let mut haystack = generator.next_haystack();

    // Generating number sequence (1, 5, 10, 15, ...) up to the estimated number of iterations/ms
    let iterations_min = opts.min_iterations_per_sample.max(1);
    let iterations_max = iterations_per_ms
        .min(opts.max_iterations_per_sample)
        .max(iterations_min);
    let mut iterations_choices = (iterations_min..=iterations_max)
        .step_by(5)
        .map(|i| 1.max(i))
        .cycle();

    let mut needles = vec![];

    for i in 0..opts.max_samples {
        // Trying not to stress benchmarking loop with to much of clock calls
        if i % iterations_per_ms == 0 && Instant::now() >= deadline {
            break;
        }
        if (i + 1) % opts.samples_per_haystack == 0 {
            haystack = generator.next_haystack();
        }

        let iterations = iterations_choices.next().unwrap();

        needles.clear();
        generator.next_needles(&haystack, iterations, &mut needles);

        // !!! IMPORTANT !!!
        // baseline and candidate should be called in different order in those two branches.
        // This equalize the probability of facing unfortunate circumstances like cache misses for both functions
        let (base_sample, candidate_sample) = if i % 2 == 0 {
            let base_sample = base.measure(&haystack, &needles);
            let candidate_sample = candidate.measure(&haystack, &needles);

            (base_sample, candidate_sample)
        } else {
            let candidate_sample = candidate.measure(&haystack, &needles);
            let base_sample = base.measure(&haystack, &needles);

            (base_sample, candidate_sample)
        };

        base_samples.push(base_sample as i64 / iterations as i64);
        candidate_samples.push(candidate_sample as i64 / iterations as i64);
    }

    if let Some(path) = samples_dump_location {
        let file_name = format!("{}-{}.csv", base.name(), candidate.name());
        let file_path = path.as_ref().join(file_name);
        write_raw_measurements(file_path, &base_samples, &candidate_samples);
    }

    let base = Summary::from(&base_samples).unwrap();
    let candidate = Summary::from(&candidate_samples).unwrap();
    let diff = base_samples
        .into_iter()
        .zip(candidate_samples)
        .map(|(b, c)| c - b)
        .collect();
    (base, candidate, diff)
}

/// Estimates the number of iterations achievable in 1 ms by given pair of functions.
///
/// If functions are to slow to be executed in 1ms, the number of iterations will be 1.
fn estimate_iterations_per_ms<H, N>(
    generator: &mut dyn Generator<Haystack = H, Needle = N>,
    a: &dyn BenchmarkFn<H, N>,
    b: &dyn BenchmarkFn<H, N>,
) -> usize {
    let mut needles = Vec::with_capacity(1);

    let haystack = generator.next_haystack();
    generator.next_needles(&haystack, 1, &mut needles);

    // Measure the amount of iterations achievable in (factor * 1ms) and later divide by this factor
    // to calculate average number of iterations per 1ms
    let factor = 10;
    let duration = Duration::from_millis(1);
    let deadline = Instant::now() + duration * factor;
    let mut iterations = 0;
    while Instant::now() < deadline {
        b.measure(&haystack, &needles);
        a.measure(&haystack, &needles);
        iterations += 1;
    }

    1.max(iterations / factor as usize)
}

pub fn calculate_run_result<N: Into<String>>(
    baseline: (N, Summary<i64>),
    candidate: (N, Summary<i64>),
    diff: Vec<i64>,
    filter_outliers: bool,
) -> RunResult {
    let n = diff.len();

    let diff_summary = if filter_outliers {
        let input = diff.to_vec();
        let (min, max) = iqr_variance_thresholds(input).unwrap_or((i64::MIN, i64::MAX));

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

    RunResult {
        base_name: baseline.0.into(),
        candidate_name: candidate.0.into(),
        baseline: baseline.1,
        candidate: candidate.1,
        diff: diff_summary,
        // significant result is far away from 0 and have more than 0.5%
        // base/candidate difference
        // z_score = 2.6 corresponds to 99% significance level
        significant: z_score.abs() >= 2.6 && (diff_summary.mean / candidate.1.mean).abs() > 0.005,
        outliers: outliers_filtered,
    }
}

/// Describes the results of a single benchmark run
pub struct RunResult {
    /// name of a baseline function
    pub base_name: String,

    /// name of a candidate function
    pub candidate_name: String,

    /// statistical summary of baseline function measurements
    pub baseline: Summary<i64>,

    /// statistical summary of candidate function measurements
    pub candidate: Summary<i64>,

    /// individual measurements of a benchmark (candidate - baseline)
    pub diff: Summary<i64>,

    /// Is difference is statistically significant
    pub significant: bool,

    /// Numbers of detected and filtered outliers
    pub outliers: usize,
}

fn write_raw_measurements(path: impl AsRef<Path>, base: &[i64], candidate: &[i64]) {
    let mut file = BufWriter::new(File::create(path).unwrap());

    for (b, c) in base.iter().zip(candidate) {
        writeln!(&mut file, "{},{}", b, c).unwrap();
    }
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

impl<'a, T: PartialOrd + Copy + Default + 'a> Summary<T> {
    pub fn from<C>(values: C) -> Option<Self>
    where
        T: ToPrimitive,
        C: IntoIterator<Item = &'a T>,
    {
        Self::running(values.into_iter().copied()).last()
    }

    pub fn running<I>(iter: I) -> impl Iterator<Item = Summary<T>>
    where
        T: ToPrimitive,
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
        let fvalue = value.to_f64().unwrap();

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
fn iqr_variance_thresholds(mut input: Vec<i64>) -> Option<(i64, i64)> {
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

    Some((input[outliers_cnt], input[input.len() - outliers_cnt]))
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
    let measures: Vec<_> = (0..iterations)
        .into_iter()
        .map(|_| target.measure(1))
        .collect();
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
    /// This test is possibly brittle. Theoretically it can fail because there is no guarantee
    /// that OS scheduler will wake up thread soon enough to meet measurement target. We try to mitigate
    /// this possibility repeating test several times and taking median as target measurement.
    #[test]
    fn check_measure_time() {
        let delay = 1;
        let mut target = benchmark_fn("foo", move || thread::sleep(Duration::from_millis(delay)));

        let median = median_execution_time(target.as_mut(), 10) / NS_TO_MS;
        assert_eq!(delay, median);
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
