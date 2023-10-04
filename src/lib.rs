use std::{collections::HashMap, hint::black_box, time::Instant};

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
    fn measure(&self, payload: &P, measurements: &mut [u64]);
}

struct Func<F> {
    pub func: F,
}

impl<F, P, O> BenchmarkFn<P, O> for Func<F>
where
    F: Fn(&P) -> O,
{
    fn measure(&self, payload: &P, measurements: &mut [u64]) {
        for time in measurements.iter_mut() {
            let start = Instant::now();
            let result = black_box((self.func)(payload));
            *time = start.elapsed().as_nanos() as u64;
            drop(result);
        }
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
    fn measure(&self, payload: &P, measurements: &mut [u64]) {
        for time in measurements.iter_mut() {
            let payload = (self.setup)(payload);
            let start = Instant::now();
            let result = black_box((self.func)(payload));
            *time = start.elapsed().as_nanos() as u64;
            drop(result);
        }
    }
}

pub trait Generator {
    type Output;
    fn next_payload(&mut self) -> Self::Output;
}

pub trait Reporter {
    fn before_start(&mut self) {}
    fn on_complete(&mut self, baseline: &str, candidate: &str, measurements: &[(u64, u64)]);
}

pub struct Benchmark<P, O> {
    payload_generator: Box<dyn Generator<Output = P>>,
    funcs: HashMap<String, Box<dyn BenchmarkFn<P, O>>>,
    iterations: usize,
}

impl<P, O> Benchmark<P, O> {
    pub fn new(generator: impl Generator<Output = P> + 'static) -> Self {
        Self {
            payload_generator: Box::new(generator),
            funcs: HashMap::new(),
            iterations: 1000,
        }
    }

    pub fn set_iterations(&mut self, iterations: usize) {
        self.iterations = iterations;
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
            self.iterations,
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
                self.iterations,
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
                self.iterations,
            );
            reporter.on_complete(baseline.as_ref(), name, measurements.as_slice());
        }
    }

    pub fn measure(
        generator: &mut dyn Generator<Output = P>,
        base: &dyn BenchmarkFn<P, O>,
        candidate: &dyn BenchmarkFn<P, O>,
        iter: usize,
    ) -> Vec<(u64, u64)> {
        let mut base_measurements = vec![0; iter];
        let mut candidate_measurements = vec![0; iter];

        // let mut batch_size = [1, 1, 25, 25, 50, 50, 75, 75, 100, 100].iter().cycle();
        // let mut batch_size = [1, 25, 50, 75, 100].iter().cycle();
        let mut batch_size = [1].iter().cycle();
        let mut offset = 0;
        let mut baseline_first = false;
        while offset < iter {
            let payload = generator.next_payload();
            let batch = *batch_size.next().unwrap();
            let batch = batch.min(iter - offset);
            baseline_first = !baseline_first;
            if baseline_first {
                base.measure(&payload, &mut base_measurements[offset..offset + batch]);
                candidate.measure(
                    &payload,
                    &mut candidate_measurements[offset..offset + batch],
                );
            } else {
                candidate.measure(
                    &payload,
                    &mut candidate_measurements[offset..offset + batch],
                );
                base.measure(&payload, &mut base_measurements[offset..offset + batch]);
            }

            offset += batch;
        }

        base_measurements
            .into_iter()
            .zip(candidate_measurements.into_iter())
            .collect()
    }

    pub fn list_functions(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(String::as_str)
    }
}
