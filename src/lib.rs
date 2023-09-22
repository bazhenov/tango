use rand::{rngs::SmallRng, Rng, SeedableRng};
use reporting::Reporter;
use std::{
    any::{type_name, TypeId},
    collections::HashMap,
    io,
    time::Instant,
};

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

            let base_min = base.iter().fold(i64::max_value(), |min, i| min.min(*i));
            let candidate_min = candidate
                .iter()
                .fold(i64::max_value(), |min, i| min.min(*i));

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
                    "{:40} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>11}",
                    "name",
                    "B min",
                    "C min",
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

pub struct Benchmark<P, O> {
    payload_generator: Box<dyn Generator<Output = P>>,
    funcs: HashMap<String, fn(&P) -> O>,
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

    pub fn add_function(&mut self, name: impl Into<String>, f: fn(&P) -> O) {
        self.funcs.insert(name.into(), f);
    }

    pub fn run_pair(
        &mut self,
        baseline: impl AsRef<str>,
        candidate: impl AsRef<str>,
        reporter: &mut impl Reporter,
    ) {
        let baseline_f = self.funcs.get(baseline.as_ref()).unwrap();
        let candidate_f = self.funcs.get(candidate.as_ref()).unwrap();

        let measurements = measure(
            self.payload_generator.as_mut(),
            *baseline_f,
            *candidate_f,
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
            let measurements = measure(self.payload_generator.as_mut(), *f, *f, self.iterations);
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
            let measurements = measure(
                self.payload_generator.as_mut(),
                *baseline_f,
                *func,
                self.iterations,
            );
            reporter.on_complete(baseline.as_ref(), name, measurements.as_slice());
        }
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

#[inline(never)]
pub fn measure<O, P>(
    generator: &mut dyn Generator<Output = P>,
    base: fn(&P) -> O,
    candidate: fn(&P) -> O,
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
        m.push(time_call(f, &payload));
    }

    base_measurements
        .into_iter()
        .zip(candidate_measurements.into_iter())
        .collect()
}

fn time_call<O, P>(f: fn(P) -> O, payload: P) -> u64 {
    let start = Instant::now();
    let result = f(payload);
    let time = start.elapsed().as_nanos() as u64;
    drop(result);
    time
}

pub struct TypedFunction<I> {
    pub type_id: TypeId,
    type_name: String,
    pub f: fn(I),
}

impl<I: 'static> TypedFunction<I> {
    pub fn new(inner: fn(I)) -> Self {
        Self {
            type_id: TypeId::of::<I>(),
            type_name: type_name::<I>().to_string(),
            f: inner,
        }
    }

    pub fn type_name(&self) -> &str {
        self.type_name.as_str()
    }

    pub fn downcast<F>(&self) -> Option<F> {
        None
    }
}
