use std::{io, time::Instant};

use rand::{rngs::SmallRng, Rng, SeedableRng};

pub trait Generator {
    type Output: ?Sized;
    fn next_payload(&mut self) -> &Self::Output;
}

#[derive(Clone)]
pub struct FixedStringGenerator {
    string: String,
}

impl Generator for FixedStringGenerator {
    type Output = str;

    fn next_payload(&mut self) -> &Self::Output {
        &self.string[..]
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
        let rng = SmallRng::seed_from_u64(42);
        Ok(Self {
            string,
            char_indicies,
            rng,
            length: 10000,
        })
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

pub fn std(s: &str) -> (u128, usize) {
    let start = Instant::now();
    let result = { s.chars().count() };
    (start.elapsed().as_nanos(), result)
}

pub fn std_count(s: &str) -> (u128, usize) {
    let start = Instant::now();
    let result = {
        let mut l = 0;
        let mut chars = s.chars();
        while chars.next().is_some() {
            l += 1;
        }
        l
    };
    (start.elapsed().as_nanos(), result)
}

pub fn std_count_rev(s: &str) -> (u128, usize) {
    let start = Instant::now();
    let result = {
        let mut l = 0;
        let mut chars = s.chars().rev();
        while chars.next().is_some() {
            l += 1;
        }
        l
    };
    (start.elapsed().as_nanos(), result)
}

pub fn std_5000(s: &str) -> (u128, usize) {
    let start = Instant::now();
    let result = { s.chars().take(5000).count() };
    (start.elapsed().as_nanos(), result)
}

pub fn std_4925(s: &str) -> (u128, usize) {
    let start = Instant::now();
    let result = { s.chars().take(4925).count() };
    (start.elapsed().as_nanos(), result)
}
