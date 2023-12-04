use std::{any::type_name, marker::PhantomData};

use crate::Generator;
use rand::{rngs::SmallRng, Fill, Rng, SeedableRng};

#[derive(Clone)]
pub struct RandomVec<T>(SmallRng, usize, PhantomData<T>, String);

impl<T> RandomVec<T> {
    #[allow(unused)]
    pub fn new(size: usize) -> Self {
        Self(
            SmallRng::seed_from_u64(42),
            size,
            PhantomData,
            format!("{}/{}", type_name::<T>(), size),
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

    fn sync(&mut self, seed: u64) {
        self.0 = SmallRng::seed_from_u64(seed);
    }
}
