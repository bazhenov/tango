use rand::{thread_rng, Rng};
use rust_pairwise_testing::{
    std, std_4925, std_5000, std_count, std_count_rev, Generator, RandomStringGenerator,
};
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    time::Instant,
};

fn main() -> io::Result<()> {
    let generator = RandomStringGenerator::new()?;

    // this trick is require for compiler not to uroll measurement loops
    let mut rand = thread_rng();
    let iter = 50000;
    let iter = rand.gen_range(iter..iter + 1);
    let iter = iter & (usize::MAX << 1);

    println!(
        "{:40} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "name", "B min", "C min", "min ∆", "B mean", "C mean", "mean ∆", "mean ∆ (%)"
    );

    let data = measure(generator.clone(), std, std, iter);
    report("std / std", data, Some("std-std.csv"))?;

    let data = measure(generator.clone(), std_count, std_count, iter);
    report(
        "std_count / std_count",
        data,
        Some("std_count-std_count.csv"),
    )?;

    let data = measure(generator.clone(), std_count_rev, std_count_rev, iter);
    report(
        "std_count_rev / std_count_rev",
        data,
        Some("std_count_rev-std_count_rev.csv"),
    )?;

    let data = measure(generator.clone(), std_5000, std_4925, iter);
    report("std_5000 / std_4925", data, Some("std_5000-std_4925.csv"))?;

    let data = measure(generator.clone(), std_count, std_count_rev, iter);
    report(
        "std_count / std_count_rev",
        data,
        Some("std_count-std_count_rev.csv"),
    )?;

    let data = measure(generator.clone(), std, std_count, iter);
    report("std / std_count", data, Some("std-std_count.csv"))?;

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

    if let Some(file) = file {
        let mut file = BufWriter::new(File::create(file)?);

        // Writing at most 5000 points to csv file. GNUplot can't handle more
        let factor = 1.max(base.len() / 5000);

        for i in 0..base.len() {
            if i % factor == 0 {
                writeln!(&mut file, "{},{}", base[i], candidate[i])?;
            }
        }
    }

    Ok(())
}

fn measure<O, G: Generator, B: Fn(&G::Output) -> O, C: Fn(&G::Output) -> O>(
    mut generator: G,
    base: B,
    candidate: C,
    iter: usize,
) -> Vec<(u64, u64)> {
    let mut result = Vec::with_capacity(iter);
    let mut rand = thread_rng();

    for _ in 0..iter {
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

        result.push((base, candidate))
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
