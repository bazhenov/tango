use rand::{rngs::SmallRng, Rng, SeedableRng};
use rust_pairwise_testing::Generator;
use std::{hint::black_box, io};

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

pub fn sum(n: usize) -> usize {
    let mut sum = 0;
    for i in 0..black_box(n) {
        sum += black_box(i);
    }
    sum
}

//#[repr(align(32))]
pub fn std(s: &String) -> usize {
    s.chars().count()
}

//#[repr(align(32))]
pub fn std_count(s: &String) -> usize {
    let mut l = 0;
    for _ in s.chars() {
        l += 1;
    }
    l
}

//#[repr(align(32))]
pub fn std_count_rev(s: &String) -> usize {
    let mut l = 0;
    for _ in s.chars().rev() {
        l += 1;
    }
    l
}

// #[repr(align(32))]
pub fn std_5000(s: &String) -> usize {
    s.chars().take(5000).count()
}

// #[repr(align(32))]
pub fn std_4925(s: &String) -> usize {
    s.chars().take(4925).count()
}
