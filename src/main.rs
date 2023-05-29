use rand::{rngs::SmallRng, thread_rng, Rng, SeedableRng};
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    time::Instant,
};

trait Generator {
    type Output: ?Sized;
    fn next_payload(&mut self) -> &Self::Output;
}

#[derive(Clone)]
struct FixedStringGenerator {
    string: String,
}

impl Generator for FixedStringGenerator {
    type Output = str;

    fn next_payload(&mut self) -> &Self::Output {
        &self.string[..]
    }
}

#[derive(Clone)]
struct RandomStringGenerator {
    string: String,
    char_indicies: Vec<usize>,
    rng: SmallRng,
    length: usize,
}

impl RandomStringGenerator {
    fn new(string: String, length: usize) -> Self {
        let char_indicies = string
            .char_indices()
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        let rng = SmallRng::seed_from_u64(42);
        Self {
            string,
            char_indicies,
            rng,
            length,
        }
    }
}

impl Generator for RandomStringGenerator {
    type Output = str;

    fn next_payload(&mut self) -> &Self::Output {
        let start = self
            .rng
            .gen_range(0..self.char_indicies.len() - self.length);

        let from = self.char_indicies[start];
        let to = self.char_indicies[start + self.length];
        return &self.string[from..to];
    }
}

fn main() -> io::Result<()> {
    let input = fs::read_to_string("./input.txt")?;
    let generator = RandomStringGenerator::new(input, 5000);

    // this trick is require for compiler not to uroll measurement loops
    let mut rand = thread_rng();
    let iter = 10000;
    let iter = rand.gen_range(iter..iter + 1);
    let iter = iter & (usize::MAX << 1);
    let factor = 10;

    println!(
        "{:40} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "name", "B min", "C min", "min ∆", "B mean", "C mean", "mean ∆", "mean ∆ (%)"
    );

    let data = measure(generator.clone(), std, std, iter, factor);
    report("std / std", data, None)?;

    let data = measure(generator.clone(), std_count, std_count, iter, factor);
    report("std_count / std_count", data, None)?;

    let data = measure(
        generator.clone(),
        std_count_rev,
        std_count_rev,
        iter,
        factor,
    );
    report("std_count_rev / std_count_rev", data, None)?;

    let data = measure(generator.clone(), std_5000, std_4925, iter, factor);
    report("std_5000 / std_4925", data, None)?;

    let data = measure(generator.clone(), std_count, std_count_rev, iter, factor);
    report("std_count / std_count_rev", data, None)?;

    let data = measure(generator.clone(), std, std_count, iter, factor);
    report("std / std_count", data, Some("./result.csv"))?;

    Ok(())
}

fn report(name: &str, input: Vec<(u64, u64)>, file: Option<&str>) -> io::Result<()> {
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
    let mut diff = base
        .iter()
        .zip(candidate.iter())
        .map(|(base, candidate)| *candidate - *base)
        .collect::<Vec<i64>>();

    mask_symmetric_outliers(&mut diff);

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
    if z_score.abs() >= 2.6 {
        print!("CHANGE DETECTED");
    }
    println!();

    let base_std_err = running_std_err(&base);
    let candidate_std_err = running_std_err(&candidate);
    let diff_std_err = running_std_err(&diff);

    if let Some(file) = file {
        let mut file = BufWriter::new(File::create(file)?);

        for i in 0..base.len() {
            writeln!(
                &mut file,
                "{},{},{},{:.2},{:.2},{:.2}",
                base[i],
                candidate[i],
                diff[i],
                base_std_err[i],
                candidate_std_err[i],
                diff_std_err[i]
            )?;
        }
    }

    Ok(())
}

fn running_std_err(input: &[i64]) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.len());
    let mut running_sum_square = 0.;
    let mut running_sum = 0;
    for (i, el) in input.iter().enumerate() {
        running_sum += *el;
        let n = (i + 1) as f64;
        let mean = running_sum as f64 / n;
        running_sum_square += (*el as f64 - mean).powi(2) as f64;
        let std_err = (running_sum_square / (i + 1) as f64).sqrt() / n.sqrt();
        output.push(std_err);
    }
    output
}

fn std(s: &str) -> usize {
    s.chars().count()
}

fn std_count(s: &str) -> usize {
    let mut l = 0;
    let mut chars = s.chars();
    while chars.next().is_some() {
        l += 1;
    }
    l
}

fn std_count_rev(s: &str) -> usize {
    let mut l = 0;
    let mut chars = s.chars().rev();
    while chars.next().is_some() {
        l += 1;
    }
    l
}

fn std_5000(s: &str) -> usize {
    s.chars().take(5000).count()
}

fn std_4925(s: &str) -> usize {
    s.chars().take(4925).count()
}

#[inline(never)]
fn measure<O, G: Generator, B: Fn(&G::Output) -> O, C: Fn(&G::Output) -> O>(
    mut generator: G,
    base: B,
    candidate: C,
    iter: usize,
    factor: usize,
) -> Vec<(u64, u64)> {
    let mut result = Vec::with_capacity(iter);
    let mut rand = thread_rng();

    for i in 0..iter {
        let payload = generator.next_payload();

        let (base, candidate) = if rand.gen_bool(0.5) {
            let base = time_call(&base, payload);
            let candidate = time_call(&candidate, payload);

            (base, candidate)
        } else {
            let candidate = time_call(&candidate, payload);
            let base = time_call(&base, payload);

            (base, candidate)
        };

        if i % factor == 0 {
            result.push((base as u64, candidate as u64));
        }
    }

    result
}

#[inline(never)]
fn time_call<O, P: Copy, F: Fn(P) -> O>(f: F, payload: P) -> u64 {
    let started = Instant::now();
    black_box(f(payload));
    started.elapsed().as_nanos() as u64
}

#[inline]
pub fn black_box<T>(dummy: T) -> T {
    unsafe {
        let ret = std::ptr::read_volatile(&dummy);
        std::mem::forget(dummy);
        ret
    }
}

fn mask_symmetric_outliers(input: &mut [i64]) {
    let mut sorted = input.iter().copied().collect::<Vec<_>>();
    sorted.sort();

    let n = sorted.len();
    let iqr = sorted[n * 75 / 100] - sorted[n * 25 / 100];

    let mut top = sorted.len() - 1;
    let mut bottom = 0;
    let mut commited_top = top;
    let mut commited_bottom = bottom;

    let median = sorted[sorted.len() / 2];

    while bottom < top {
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

        let top_removed = sorted.len() - top - 1;
        let bottom_removed = bottom;
        let abs_diff = top_removed.abs_diff(bottom_removed);
        let deviation = abs_diff as f64 / (bottom_removed + top_removed) as f64;
        if abs_diff < 3 || deviation < 0.3 {
            commited_top = top;
            commited_bottom = bottom;
        }
    }

    for el in input.iter_mut() {
        if *el < sorted[commited_bottom] {
            *el = sorted[commited_bottom];
        } else if *el > sorted[commited_top] {
            *el = sorted[commited_top];
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mask_symmetric_outliers;

    #[test]
    fn test_symmetric_outliers() {
        let mut input = vec![-1000, 3, 3, 3, 1000];

        mask_symmetric_outliers(&mut input);

        assert_eq!(input, vec![3, 3, 3, 3, 3]);
    }
}
