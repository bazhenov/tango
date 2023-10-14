use std::{
    collections::HashMap,
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use timer::{ActiveTimer, Timer};

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
        let start = ActiveTimer::start();
        let result = black_box((self.func)(payload));
        let time = ActiveTimer::stop(start);
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
        let start = ActiveTimer::start();
        let result = black_box((self.func)(payload));
        let time = ActiveTimer::stop(start);
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
    fn on_complete(&mut self, name: &str, measurements: &[(u64, u64)]);
}

#[derive(Copy, Clone)]
pub enum RunMode {
    Iterations(NonZeroUsize),
    Time(Duration),
}

pub struct Benchmark<P, O> {
    payload_generator: Box<dyn Generator<Output = P>>,
    funcs: HashMap<String, (Box<dyn BenchmarkFn<P, O>>, Box<dyn BenchmarkFn<P, O>>)>,
    run_mode: RunMode,
    measurements_dir: Option<PathBuf>,
    reporters: Vec<Box<dyn Reporter>>,
}

impl<P, O> Benchmark<P, O> {
    pub fn new(generator: impl Generator<Output = P> + 'static) -> Self {
        Self {
            payload_generator: Box::new(generator),
            funcs: HashMap::new(),
            run_mode: RunMode::Time(Duration::from_millis(100)),
            measurements_dir: None,
            reporters: vec![],
        }
    }

    /// Sets a directory location for CSV dump of individual mesurements
    ///
    /// The format is as follows
    /// ```no_run
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
        name: impl Into<String>,
        baseline: impl BenchmarkFn<P, O> + 'static,
        candidate: impl BenchmarkFn<P, O> + 'static,
    ) {
        let key = name.into();
        assert!(!key.is_empty());
        self.funcs
            .insert(key, (Box::new(baseline), Box::new(candidate)));
    }

    pub fn run_by_name(&mut self, name: impl AsRef<str>) {
        for (key, (baseline, candidate)) in &self.funcs {
            if key.contains(name.as_ref()) {
                let measurements = Self::measure(
                    self.payload_generator.as_mut(),
                    baseline.as_ref(),
                    candidate.as_ref(),
                    self.run_mode,
                    dump_location(key.as_ref(), self.measurements_dir.as_ref()),
                );
                for reporter in self.reporters.iter_mut() {
                    reporter.before_start();
                    reporter.on_complete(key, measurements.as_slice());
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

        if let Some(path) = dump_path {
            let mut file = BufWriter::new(File::create(path).unwrap());

            for (b, c) in base_measurements.iter().zip(candidate_measurements.iter()) {
                writeln!(&mut file, "{},{}", b, c).unwrap();
            }
        }

        base_measurements
            .into_iter()
            .zip(candidate_measurements)
            .collect()
    }

    pub fn list_functions(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(String::as_str)
    }

    fn set_run_mode(&mut self, run_mode: RunMode) {
        self.run_mode = run_mode;
    }
}

fn dump_location(name: &str, dir: Option<impl AsRef<Path>>) -> Option<impl AsRef<Path>> {
    dir.map(|p| p.as_ref().join(format!("{}.csv", name)))
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
