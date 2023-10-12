use std::{
    collections::HashMap,
    hint::black_box,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

use rand::{rngs::SmallRng, RngCore};

pub mod cli;

pub fn benchmark_fn<P, O, F: Fn(&P) -> O>(func: F) -> impl BenchmarkFn<P, O> {
    Func { func }
}

pub fn benchmark_fn_with_setup<P, O, I, F: Fn(I) -> O, S: Fn(&P) -> I>(
    func: F,
    setup: S,
) -> impl BenchmarkFn<P, O> {
    SetupFunc { func, setup }
}

pub trait BenchmarkFn<P, O> {
    fn measure(&self, payload: &P) -> u64;
}

struct Func<F> {
    pub func: F,
}

impl<F, P, O> BenchmarkFn<P, O> for Func<F>
where
    F: Fn(&P) -> O,
{
    fn measure(&self, payload: &P) -> u64 {
        let start = Instant::now();
        let result = black_box((self.func)(payload));
        let time = start.elapsed().as_nanos() as u64;
        drop(result);
        time
    }
}

struct SetupFunc<S, F> {
    setup: S,
    func: F,
}

impl<S, F, P, I, O> BenchmarkFn<P, O> for SetupFunc<S, F>
where
    S: Fn(&P) -> I,
    F: Fn(I) -> O,
{
    fn measure(&self, payload: &P) -> u64 {
        let payload = (self.setup)(payload);
        let start = Instant::now();
        let result = black_box((self.func)(payload));
        let time = start.elapsed().as_nanos() as u64;
        drop(result);
        time
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
    fn before_start(&mut self) {}
    fn on_complete(&mut self, baseline: &str, candidate: &str, measurements: &[(u64, u64)]);
}

#[derive(Copy, Clone)]
pub enum RunMode {
    Iterations(NonZeroUsize),
    Time(Duration),
}

pub struct Benchmark<P, O> {
    payload_generator: Box<dyn Generator<Output = P>>,
    funcs: HashMap<String, Box<dyn BenchmarkFn<P, O>>>,
    run_mode: RunMode,
}

impl<P, O> Benchmark<P, O> {
    pub fn new(generator: impl Generator<Output = P> + 'static) -> Self {
        Self {
            payload_generator: Box::new(generator),
            funcs: HashMap::new(),
            run_mode: RunMode::Time(Duration::from_millis(100)),
        }
    }

    pub fn add_function(&mut self, name: impl Into<String>, f: impl BenchmarkFn<P, O> + 'static) {
        self.funcs.insert(name.into(), Box::new(f));
    }

    pub fn run_pair(
        &mut self,
        baseline: impl AsRef<str>,
        candidate: impl AsRef<str>,
        reporter: &mut dyn Reporter,
    ) {
        let baseline_f = self.funcs.get(baseline.as_ref()).unwrap();
        let candidate_f = self.funcs.get(candidate.as_ref()).unwrap();

        let measurements = Self::measure(
            self.payload_generator.as_mut(),
            baseline_f.as_ref(),
            candidate_f.as_ref(),
            self.run_mode,
        );
        reporter.before_start();
        reporter.on_complete(
            baseline.as_ref(),
            candidate.as_ref(),
            measurements.as_slice(),
        );
    }

    pub fn run_calibration(&mut self, reporter: &mut dyn Reporter) {
        reporter.before_start();
        for (name, f) in self.funcs.iter() {
            let measurements = Self::measure(
                self.payload_generator.as_mut(),
                f.as_ref(),
                f.as_ref(),
                self.run_mode,
            );
            reporter.on_complete(name, name, measurements.as_slice());
        }
    }

    pub fn run_all_against(&mut self, baseline: impl AsRef<str>, reporter: &mut impl Reporter) {
        let baseline_f = self.funcs.get(baseline.as_ref()).unwrap();
        let mut candidates = self
            .funcs
            .iter()
            .filter(|(name, _)| *name != baseline.as_ref())
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| a.0.cmp(b.0));

        reporter.before_start();
        for (name, func) in candidates {
            let measurements = Self::measure(
                self.payload_generator.as_mut(),
                baseline_f.as_ref(),
                func.as_ref(),
                self.run_mode,
            );
            reporter.on_complete(baseline.as_ref(), name, measurements.as_slice());
        }
    }

    pub fn measure(
        generator: &mut dyn Generator<Output = P>,
        base: &dyn BenchmarkFn<P, O>,
        candidate: &dyn BenchmarkFn<P, O>,
        run_mode: RunMode,
    ) -> Vec<(u64, u64)> {
        let mut base_measurements = vec![];
        let mut candidate_measurements = vec![];

        match run_mode {
            RunMode::Iterations(iter) => {
                for i in 0..usize::from(iter) {
                    let payload = generator.next_payload();
                    if i % 2 == 0 {
                        base_measurements.push(base.measure(&payload));
                        candidate_measurements.push(candidate.measure(&payload));
                    } else {
                        candidate_measurements.push(candidate.measure(&payload));
                        base_measurements.push(base.measure(&payload));
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
                        base_measurements.push(base.measure(&payload));
                        candidate_measurements.push(candidate.measure(&payload));
                    } else {
                        candidate_measurements.push(candidate.measure(&payload));
                        base_measurements.push(base.measure(&payload));
                    }
                }
            }
        }

        base_measurements
            .into_iter()
            .zip(candidate_measurements.into_iter())
            .collect()
    }

    pub fn list_functions(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(String::as_str)
    }

    fn set_run_mode(&mut self, run_mode: RunMode) {
        self.run_mode = run_mode;
    }
}

/// Running variance iterator
///
/// Provides a running (streaming variance) for a given iterator of observations.
/// Uses simple variance formula: `Var(X) = E[X^2] - E[X]^2`.
struct RunningVariance<T> {
    iter: T,
    sum: f64,
    sum_of_squares: f64,
    n: f64,
}

impl<T: Iterator<Item = i64>> Iterator for RunningVariance<T> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let i = f64::from(self.iter.next()? as f64);
        self.sum += i;
        self.sum_of_squares += i.powi(2);
        self.n += 1.;

        Some((self.sum_of_squares / self.n) - (self.sum / self.n).powi(2))
    }
}

impl<I, T: Iterator<Item = I>> From<T> for RunningVariance<T> {
    fn from(value: T) -> Self {
        Self {
            iter: value,
            sum: 0.,
            sum_of_squares: 0.,
            n: 0.,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, SeedableRng};

    struct RngIterator(SmallRng);

    impl Iterator for RngIterator {
        type Item = u32;

        fn next(&mut self) -> Option<Self::Item> {
            Some(self.0.next_u32())
        }
    }

    #[test]
    fn check_running_variance() {
        let input = [1i64, 2, 3, 4, 5, 6, 7];
        let variances = RunningVariance::from(input.into_iter());
        let expected = &[0., 0.25, 0.666, 1.25, 2.0, 2.916];

        for (value, expected_value) in variances.zip(expected.iter()) {
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

    #[test]
    fn check_filter_outliers() {
        let input = vec![
            1i64, -2, 3, -4, 5, -6, 7, -8, 9, -10, //
            101, -102, 103, -104, 105, -106, 107, -108, 109, -110,
        ];

        let (min, max) = outliers_threshold(input).unwrap();
        assert!(min < 1, "Minimum is: {}", min);
        assert!(10 < max && max <= 101, "Maximum is: {}", max);
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
    input.sort_by(|a, b| a.abs().cmp(&b));
    let variance = RunningVariance::from(input.iter().copied());

    let mut value_and_variance = input
        .iter()
        .copied()
        .zip(variance)
        .skip(input.len() * 30 / 100); // Looking only 30% topmost values

    let mut prev_variance = 0.;
    while let Some((value, var)) = value_and_variance.next() {
        if prev_variance > 0. {
            if var / prev_variance > 2. {
                return Some((-value.abs(), value.abs()));
            }
        }
        prev_variance = var;
    }

    None
}
