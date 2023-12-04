use crate::Generator;
use rand::{rngs::SmallRng, Rng, SeedableRng};

pub struct RandomVec {
    size: usize,
    rng: SmallRng,
}

impl RandomVec {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            rng: SmallRng::from_entropy(),
        }
    }
}

impl Generator for RandomVec {
    type Haystack = Vec<u32>;
    type Needle = ();

    fn next_haystack(&mut self) -> Self::Haystack {
        let mut v = vec![0; self.size];
        self.rng.fill(&mut v[..]);
        v
    }

    fn next_needle(&mut self, _: &Self::Haystack) -> Self::Needle {
        todo!()
    }

    fn reset(&mut self, seed: u64) {
        self.rng = SmallRng::seed_from_u64(seed)
    }
}
