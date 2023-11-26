use rand::{rngs::SmallRng, Fill, Rng, SeedableRng};
use std::{any::type_name, hint::black_box, io, marker::PhantomData};
use tango_bench::Generator;

/// HTML page with a lot of chinese text to test UTF8 decoding speed
const INPUT_TEXT: &str = include_str!("./input.txt");

#[derive(Clone)]
pub struct RandomVec<T>(SmallRng, usize, PhantomData<T>);

impl<T> RandomVec<T> {
    #[allow(unused)]
    pub fn new(size: usize) -> Self {
        Self(SmallRng::seed_from_u64(42), size, PhantomData)
    }
}

impl<T: Default + Copy> Generator for RandomVec<T>
where
    [T]: Fill,
{
    type Haystack = Vec<T>;
    type Needle = ();

    fn next_haystack(&mut self) -> Self::Haystack {
        let RandomVec(rng, size, _) = self;
        let mut v = vec![T::default(); *size];
        rng.fill(&mut v[..]);
        v
    }

    fn next_needle(&mut self, _haystack: &Self::Haystack) -> Self::Needle {}

    fn name(&self) -> String {
        format!("RandomVec<{}, {}>", type_name::<T>(), self.1)
    }

    fn reset(&mut self) {}
}

#[derive(Clone)]
pub struct RandomString {
    char_indicies: Vec<usize>,
    rng: SmallRng,
    length: usize,
}

impl RandomString {
    #[allow(unused)]
    pub fn new() -> io::Result<Self> {
        let char_indicies = INPUT_TEXT
            .char_indices()
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        let rng = SmallRng::seed_from_u64(42);
        Ok(Self {
            char_indicies,
            rng,
            length: 50000,
        })
    }
}
impl Generator for RandomString {
    type Haystack = String;
    type Needle = ();

    fn next_haystack(&mut self) -> Self::Haystack {
        let start = self
            .rng
            .gen_range(0..self.char_indicies.len() - self.length);

        let from = self.char_indicies[start];
        let to = self.char_indicies[start + self.length];
        INPUT_TEXT[from..to].to_string()
    }

    fn name(&self) -> String {
        format!("RandomString<{}>", self.length)
    }

    fn next_needle(&mut self, _haystack: &Self::Haystack) -> Self::Needle {}

    fn reset(&mut self) {
        self.rng = SmallRng::seed_from_u64(42);
    }
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(unused)]
pub fn sum(n: usize) -> usize {
    let mut sum = 0;
    for i in 0..black_box(n) {
        sum += black_box(i);
    }
    sum
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(unused)]
pub fn factorial(mut n: usize) -> usize {
    let mut result = 1usize;
    while n > 0 {
        result = result.wrapping_mul(black_box(n));
        n -= 1;
    }
    result
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(unused)]
#[allow(clippy::ptr_arg)]
pub fn str_std<T>(s: &String, _: &T) -> usize {
    s.chars().count()
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(unused)]
#[allow(clippy::ptr_arg)]
pub fn str_count<T>(s: &String, _: &T) -> usize {
    let mut l = 0;
    for _ in s.chars() {
        l += 1;
    }
    l
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(unused)]
#[allow(clippy::ptr_arg)]
pub fn str_count_rev<T>(s: &String, _: &T) -> usize {
    let mut l = 0;
    for _ in s.chars().rev() {
        l += 1;
    }
    l
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(unused)]
#[allow(clippy::ptr_arg)]
pub fn str_take<T>(n: usize, s: &String, _: &T) -> usize {
    s.chars().take(black_box(n)).count()
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
pub fn sort_unstable<T: Ord + Copy, N>(input: &Vec<T>, _: &N) -> T {
    let mut input = input.clone();
    input.sort_unstable();
    input[input.len() / 2]
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
pub fn sort_stable<T: Ord + Copy, N>(input: &Vec<T>, _: &N) -> T {
    let mut input = input.clone();
    input.sort();
    input[input.len() / 2]
}
