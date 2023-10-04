use rand::{rngs::SmallRng, Rng, SeedableRng};
use reporting::Reporter;
use std::{collections::HashMap, io, time::Instant};

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

pub struct Func<F> {
    pub func: F,
}

impl<F, P, O> BenchmarkFn<P, O> for Func<F>
where
    F: Fn(&P) -> O,
{
    fn measure(&self, payload: &P, measurements: &mut [u64]) {
        for time in measurements.iter_mut() {
            let start = Instant::now();
            let result = (self.func)(payload);
            *time = start.elapsed().as_nanos() as u64;
            drop(result);
        }
    }
}

pub struct SetupFunc<S, F> {
    pub setup: S,
    pub func: F,
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
            let result = (self.func)(payload);
            *time = start.elapsed().as_nanos() as u64;
            drop(result);
        }
    }
}

pub trait Generator {
    type Output;
    fn next_payload(&mut self) -> Self::Output;
}

#[derive(Clone)]
pub struct FixedStringGenerator {
    string: String,
}

impl Generator for FixedStringGenerator {
    type Output = String;

    fn next_payload(&mut self) -> Self::Output {
        self.string.clone()
    }
}

#[derive(Clone)]
pub struct RandomStringGenerator {
    string: String,
    char_indicies: Vec<usize>,
    rng: SmallRng,
    length: usize,
}

impl RandomStringGenerator {
    pub fn new() -> io::Result<Self> {
        let string = std::fs::read_to_string("./input.txt")?;
        let char_indicies = string
            .char_indices()
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        let rng = SmallRng::from_entropy();
        Ok(Self {
            string,
            char_indicies,
            rng,
            length: 50000,
        })
    }
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
        reporter: &mut impl Reporter,
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

    pub fn run_calibration(&mut self, reporter: &mut impl Reporter) {
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
        let mut base_measurements = Vec::with_capacity(iter);
        let mut candidate_measurements = Vec::with_capacity(iter);

        for i in 0..iter * 2 {
            let payload = generator.next_payload();

            let baseline = i % 2 == 0;
            let (f, m) = if baseline {
                (base, &mut base_measurements)
            } else {
                (candidate, &mut candidate_measurements)
            };
            m.push(0);
            let len = m.len();
            f.measure(&payload, &mut m[len - 1..len]);
        }

        base_measurements
            .into_iter()
            .zip(candidate_measurements.into_iter())
            .collect()
    }
}

impl Generator for RandomStringGenerator {
    type Output = String;

    fn next_payload(&mut self) -> Self::Output {
        let start = self
            .rng
            .gen_range(0..self.char_indicies.len() - self.length);

        let from = self.char_indicies[start];
        let to = self.char_indicies[start + self.length];
        self.string[from..to].to_string()
    }
}

pub mod reporting {
    use std::io::Write;
    use std::{fs::File, io::BufWriter};

    pub trait Reporter {
        fn before_start(&mut self) {}
        fn on_complete(&mut self, baseline: &str, candidate: &str, measurements: &[(u64, u64)]);
    }

    #[derive(Default)]
    pub struct ConsoleReporter {
        header_printed: bool,
        write_data: bool,
    }

    impl ConsoleReporter {
        pub fn set_write_data(&mut self, write_data: bool) {
            self.write_data = write_data;
        }
    }

    impl Reporter for ConsoleReporter {
        fn on_complete(&mut self, baseline_name: &str, candidate_name: &str, input: &[(u64, u64)]) {
            let name = format!("{} / {}", baseline_name, candidate_name);
            let base = input
                .iter()
                .map(|(base, _)| *base as i64)
                .collect::<Vec<_>>();
            let candidate = input
                .iter()
                .map(|(_, candidate)| *candidate as i64)
                .collect::<Vec<_>>();

            let base_min = *base.iter().min().unwrap();
            let candidate_min = *candidate.iter().min().unwrap();

            let base_max = *base.iter().max().unwrap();
            let candidate_max = *candidate.iter().max().unwrap();

            let n = base.len() as f64;

            let base_mean = base.iter().sum::<i64>() as f64 / n;
            let candidate_mean = candidate.iter().sum::<i64>() as f64 / n;
            let mut diff = input
                .iter()
                .map(|(base, candidate)| *candidate as i64 - *base as i64)
                .collect::<Vec<i64>>();

            let filtered = mask_symmetric_outliers(&mut diff);

            let diff_mean = diff.iter().sum::<i64>() as f64 / n;
            let variance = diff
                .iter()
                .map(|i| (*i as f64 - diff_mean).powi(2))
                .sum::<f64>()
                / (n - 1.);
            let std_dev = variance.sqrt();
            let std_err = std_dev / n.sqrt();
            let z_score = diff_mean / std_err;

            print!("{:40} ", name);
            print!("{:10} {:10} ", base_min, candidate_min);
            print!("{:10} {:10} ", base_max, candidate_max);
            let min_diff = (candidate_min - base_min) as f64 / base_min as f64 * 100.;
            print!("{:9.1}% ", min_diff);
            print!("{:10.1} {:10.1} ", base_mean, candidate_mean);
            print!("{:10.1} ", diff_mean);
            print!("{:9.1}% ", diff_mean / base_mean * 100.);
            print!(
                "{:5} {:4.1}% ",
                filtered,
                filtered as f64 / (n as f64) * 100.
            );
            if z_score.abs() >= 2.6 {
                if diff_mean > 0. {
                    print!("CANDIDATE SLOWER");
                } else {
                    print!("CANDIDATE FASTER");
                }
            }
            println!();

            if self.write_data {
                let file_name = format!("{}-{}.csv", baseline_name, candidate_name);
                let mut file = BufWriter::new(File::create(file_name).unwrap());

                // Writing at most 1000 points to csv file. GNUplot can't handle more
                let factor = 1.max(base.len() / 1000);

                for i in 0..base.len() {
                    if i % factor == 0 {
                        writeln!(&mut file, "{},{}", base[i], candidate[i]).unwrap();
                    }
                }
            }
        }

        fn before_start(&mut self) {
            if !self.header_printed {
                self.header_printed = true;
                println!(
                    "{:40} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>11}",
                    "name",
                    "B min",
                    "C min",
                    "B max",
                    "C max",
                    "min ∆",
                    "B mean",
                    "C mean",
                    "mean ∆",
                    "mean ∆ (%)",
                    "outliers"
                );
            }
        }
    }

    /// Winsorizing symmetric outliers in a slice
    ///
    /// [Winsorizing][winsorize] is a tchinque of removing outliers in a dataset effectively masking then
    /// with what the most exteme observations left (wo. outliers). This particular algorithm will remove outliers
    /// only if following criteria holds:
    ///
    /// - only 5% of observations are removed from each size
    /// - only outliers greater than 3 IQR from median are removed
    ///
    /// [winsorize]: https://en.wikipedia.org/wiki/Winsorizing
    fn mask_symmetric_outliers(input: &mut [i64]) -> usize {
        let mut filtered = 0;
        let n = input.len();

        let mut sorted = input.to_vec();
        sorted.sort();

        let iqr = sorted[n * 75 / 100] - sorted[n * 25 / 100];

        let mut top = sorted.len() - 1;
        let mut bottom = 0;
        let mut commited_top = top;
        let mut commited_bottom = bottom;

        let median = sorted[sorted.len() / 2];

        while bottom < n * 10 / 100 && top > n * 90 / 100 {
            let bottom_diff = median - sorted[bottom];
            let top_diff = sorted[top] - median;

            let diff = bottom_diff.max(top_diff);
            if diff < 3 * iqr {
                break;
            }

            if top_diff > bottom_diff {
                top -= 1;
            } else {
                bottom += 1;
            }

            let top_removed = n - top - 1;
            let bottom_removed = bottom;
            let abs_diff = top_removed.abs_diff(bottom_removed);

            // TODO Replace this with binomial coefficient/normal distribution approximation calculations
            let deviation = abs_diff as f64 / (bottom_removed + top_removed) as f64;
            if abs_diff < 5 || deviation < 0.3 {
                commited_top = top;
                commited_bottom = bottom;
            }
        }

        for el in input.iter_mut() {
            if *el < sorted[commited_bottom] {
                *el = sorted[commited_bottom];
                filtered += 1;
            } else if *el > sorted[commited_top] {
                *el = sorted[commited_top];
                filtered += 1;
            }
        }

        filtered
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_symmetric_outliers() {
            let mut input = [50i64; 100];
            input[0] = -1000;
            input[1] = -1000;

            mask_symmetric_outliers(&mut input);

            assert_eq!(input[0], 50);
            assert_eq!(input[input.len() - 1], 50);
        }
    }
}
