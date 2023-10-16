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
    run_mode: RunMode,
    measurements_dir: Option<PathBuf>,
    reporters: Vec<Box<dyn Reporter>>,
}

impl<P, O> Default for Benchmark<P, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P, O> Benchmark<P, O> {
    pub fn new() -> Self {
        Self {
            funcs: BTreeMap::new(),
            run_mode: RunMode::Time(Duration::from_millis(100)),
            measurements_dir: None,
            reporters: vec![],
        }
    }

    /// Sets a directory location for CSV dump of individual mesurements
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
    pub fn set_measurements_dir(&mut self, dir: Option<impl AsRef<Path>>) {
        self.measurements_dir = dir.map(|l| l.as_ref().into())
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

    pub fn run_by_name(&mut self, payloads: &mut dyn Generator<Output = P>, name: impl AsRef<str>) {
        for (key, (baseline, candidate)) in &self.funcs {
            if key.contains(name.as_ref()) {
                let (base_summary, candidate_summary, diff) = Self::measure(
                    payloads,
                    baseline.as_ref(),
                    candidate.as_ref(),
                    self.run_mode,
                    dump_location(key.as_ref(), self.measurements_dir.as_ref()),
                );
                let results = RunResults {
                    base_name: baseline.name().to_owned(),
                    candidate_name: candidate.name().to_owned(),
                    base: base_summary,
                    candidate: candidate_summary,
                    measurements: diff,
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

    pub fn measure(
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

    fn set_run_mode(&mut self, run_mode: RunMode) {
        self.run_mode = run_mode;
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
    pub measurements: Vec<i64>,
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
        let mut iter = values.into_iter();
        let head = *iter.next()?;
        let mut min = head;
        let mut max = head;
        let mut mean = i64::from(head) as f64;
        let mut s = 0.;
        let mut n = 1;
        while let Some(value) = iter.next() {
            n += 1;
            max = max.max(*value);
            min = min.min(*value);

            let value = i64::from(*value) as f64;
            let mean_p = mean;
            mean += (value - mean) / n as f64;
            s += (value - mean_p) * (value - mean);
        }

        let variance = s / (n - 1) as f64;
        Some(Self {
            n,
            min,
            max,
            mean,
            variance,
        })
    }
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
    use std::{
        iter::Sum,
        ops::{AddAssign, Mul, Sub},
    };

    use super::*;

    #[test]
    fn check_summary_statistics() {
        let stat = Summary::from(&vec![1, 1, 2, 4]).unwrap();
        assert_eq!(stat.min, 1);
        assert_eq!(stat.max, 4);
        assert_eq!(stat.variance, 2.);

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
            assert_eq!(stat.mean, expected_mean);
            assert_eq!(stat.variance, expected_variance);
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
