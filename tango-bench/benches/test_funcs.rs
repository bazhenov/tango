use rand::{rngs::SmallRng, Fill, Rng, SeedableRng};
use std::{any::type_name, hint::black_box, marker::PhantomData, ops::Range, rc::Rc};
use tango_bench::Generator;

/// HTML page with a lot of chinese text to test UTF8 decoding speed
const INPUT_TEXT: &str = include_str!("./input.txt");

#[derive(Clone)]
pub struct RandomVec<T>(SmallRng, usize, PhantomData<T>, String);

impl<T> RandomVec<T> {
    #[allow(unused)]
    pub fn new(size: usize) -> Self {
        Self(
            SmallRng::seed_from_u64(42),
            size,
            PhantomData,
            format!("RandomVec<{}, {}>", type_name::<T>(), size),
        )
    }
}

impl<T: Default + Copy> Generator for RandomVec<T>
where
    [T]: Fill,
{
    type Haystack = Vec<T>;
    type Needle = ();

    fn next_haystack(&mut self) -> Self::Haystack {
        let RandomVec(rng, size, _, _) = self;
        let mut v = vec![T::default(); *size];
        rng.fill(&mut v[..]);
        v
    }

    fn next_needle(&mut self, _haystack: &Self::Haystack) -> Self::Needle {}

    fn name(&self) -> &str {
        &self.3
    }

    fn reset(&mut self) {}
}

#[derive(Clone)]
pub struct RandomSubstring {
    char_indicies: Vec<usize>,
    rng: SmallRng,
    length: usize,
    value: Rc<String>,
    name: String,
}

impl RandomSubstring {
    #[allow(unused)]
    pub fn new() -> Self {
        let char_indicies = INPUT_TEXT
            .char_indices()
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        let rng = SmallRng::seed_from_u64(42);
        let length = 50000;
        Self {
            char_indicies,
            rng,
            value: Rc::new(INPUT_TEXT.to_string()),
            length,
            name: format!("RandomString<{}>", length),
        }
    }
}
impl Generator for RandomSubstring {
    type Haystack = Rc<String>;
    type Needle = Range<usize>;

    fn next_haystack(&mut self) -> Self::Haystack {
        Rc::clone(&self.value)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn next_needle(&mut self, _haystack: &Self::Haystack) -> Self::Needle {
        let start = self
            .rng
            .gen_range(0..self.char_indicies.len() - self.length);

        let from = self.char_indicies[start];
        let to = self.char_indicies[start + self.length];
        from..to
    }

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
pub fn str_count<T: AsRef<String>>(s: &T, idx: &Range<usize>) -> usize {
    let mut l = 0;
    for _ in s.as_ref()[idx.start..idx.end].chars() {
        l += 1;
    }
    l
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(unused)]
#[allow(clippy::ptr_arg)]
pub fn str_count_rev<T: AsRef<String>>(s: &T, idx: &Range<usize>) -> usize {
    let mut l = 0;
    for _ in s.as_ref()[idx.start..idx.end].chars().rev() {
        l += 1;
    }
    l
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(unused)]
#[allow(clippy::ptr_arg)]
pub fn str_take(n: usize, s: &String, idx: &Range<usize>) -> usize {
    s[idx.start..idx.end].chars().take(black_box(n)).count()
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(unused)]
pub fn sort_unstable<T: Ord + Copy, N>(input: &Vec<T>, _: &N) -> T {
    let mut input = input.clone();
    input.sort_unstable();
    input[input.len() / 2]
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
#[allow(unused)]
pub fn sort_stable<T: Ord + Copy, N>(input: &Vec<T>, _: &N) -> T {
    let mut input = input.clone();
    input.sort();
    input[input.len() / 2]
}
