use std::{
    env,
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use tango::Summary;

fn main() {
    let file_name = env::args().nth(1).unwrap();
    let file = BufReader::new(File::open(file_name).unwrap());

    let mut data = vec![];
    for line in file.lines() {
        let line = line.unwrap();
        let parts = line.split(',').collect::<Vec<_>>();
        let diff = parts[1].parse::<i64>().unwrap() - parts[0].parse::<i64>().unwrap();
        data.push(diff);
    }
    let diff_summary = Summary::from(&data).unwrap();
    {
        let std_dev = diff_summary.variance.sqrt();
        let std_err = std_dev / (diff_summary.n as f64).sqrt();
        let z_score = diff_summary.mean / std_err;
        let (min, max) = confidence_intervals(data.as_slice());

        println!(
            "Original - Mean: {:.3} z-score: {:.2}, interval: [{:.2}, {:.2}]",
            diff_summary.mean, z_score, min, max
        );
    }

    let start = Instant::now();
    let rounds = 100;
    let mut means = Vec::with_capacity(rounds);
    let mut rand = SmallRng::seed_from_u64(42);
    for _ in 0..rounds {
        means.push(bootstrap(&data, &mut rand).mean);
    }
    let summary = Summary::from(&means).unwrap();

    let std_dev = summary.variance.sqrt();
    let std_err = std_dev / (summary.n as f64).sqrt();
    let z_score = summary.mean / std_err;

    // println!("{:?}", &means[0..10]);

    let (min, max) = confidence_intervals(means.as_slice());
    println!(
        "Bootstraping Mean: {:.3} z-score: {:.2}, interval: [{:.2}, {:.2}]",
        summary.mean, z_score, min, max
    );
    println!("time: {:?}", start.elapsed(),)
}

fn bootstrap(input: &[i64], rand: &mut impl Rng) -> Summary<i64> {
    let n = input.len();
    let mut v = Vec::with_capacity(input.len());
    for _ in 0..n {
        v.push(*input.choose(rand).unwrap());
    }
    Summary::from(&v).unwrap()
}

fn confidence_intervals<T: PartialOrd + Copy>(input: &[T]) -> (T, T) {
    let mut input = input.to_vec();
    input.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let l = (input.len() * 5) / 100;
    let h = (input.len() * 95) / 100;
    (input[l], input[h])
}
