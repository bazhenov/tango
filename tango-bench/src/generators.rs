use rand::{rngs::SmallRng, Fill, Rng, SeedableRng};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct RandomVec<T>(SmallRng, usize, PhantomData<T>);

impl<T> RandomVec<T> {
    #[allow(unused)]
    pub fn new(size: usize) -> Self {
        Self(SmallRng::seed_from_u64(42), size, PhantomData)
    }
}

impl<T: Default + Copy> RandomVec<T>
where
    [T]: Fill,
{
    pub fn next_haystack(&mut self) -> Vec<T> {
        let RandomVec(rng, size, _) = self;
        let mut v = vec![T::default(); *size];
        rng.fill(&mut v[..]);
        v
    }

    pub fn sync(&mut self, seed: u64) {
        self.0 = SmallRng::seed_from_u64(seed);
    }
}
