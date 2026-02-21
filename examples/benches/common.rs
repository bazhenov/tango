extern crate tango_bench;

use std::{any::type_name, convert::TryFrom, fmt::Debug, iter, marker::PhantomData, usize};
use tango_bench::{benchmark_fn, IntoBenchmarks, MeasurementSettings, DEFAULT_SETTINGS};

const SIZES: [usize; 14] = [
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];

struct Lcg(usize);

impl Lcg {
    fn next<T: TryFrom<usize>>(&mut self, max_value: usize) -> T {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        T::try_from((self.0 >> 32) % max_value).ok().unwrap()
    }
}

pub struct RandomCollection<C: FromSortedVec> {
    rng: Lcg,
    size: usize,
    value_dup_factor: usize,
    phantom: PhantomData<C>,
}

impl<C: FromSortedVec> RandomCollection<C>
where
    C::Item: Ord + Copy + TryFrom<usize>,
{
    pub fn new(size: usize, value_dup_factor: usize, seed: u64) -> Self {
        Self {
            rng: Lcg(seed as usize),
            size,
            value_dup_factor,
            phantom: PhantomData,
        }
    }
}

impl<C: FromSortedVec> RandomCollection<C>
where
    C::Item: Ord + Copy + TryFrom<usize> + Debug,
    usize: TryFrom<C::Item>,
{
    fn random_collection(&mut self) -> Sample<C> {
        let vec = generate_sorted_vec(self.size, self.value_dup_factor);
        let max = usize::try_from(*vec.last().unwrap()).ok().unwrap();

        Sample {
            collection: C::from_sorted_vec(vec),
            max_value: max,
        }
    }

    fn next_needle(&mut self, sample: &Sample<C>) -> C::Item {
        self.rng.next(sample.max_value + 1)
    }
}

fn generate_sorted_vec<T>(size: usize, dup_factor: usize) -> Vec<T>
where
    T: Ord + Copy + TryFrom<usize>,
{
    (0..)
        .map(|v| 2 * v)
        .map(|v| T::try_from(v))
        .map_while(Result::ok)
        .flat_map(|v| {
            #[allow(clippy::manual_repeat_n)]
            iter::repeat(v).take(dup_factor)
        })
        .take(size)
        .collect()
}

pub struct Sample<C> {
    collection: C,
    max_value: usize,
}

impl<C> AsRef<C> for Sample<C> {
    fn as_ref(&self) -> &C {
        &self.collection
    }
}

pub trait FromSortedVec {
    type Item;
    fn from_sorted_vec(v: Vec<Self::Item>) -> Self;
}

impl<T> FromSortedVec for Vec<T> {
    type Item = T;

    fn from_sorted_vec(v: Vec<T>) -> Self {
        v
    }
}

/// Generate benchmarks for searching in a collection.
pub fn search_benchmarks<C, F>(f: F) -> impl IntoBenchmarks
where
    C: FromSortedVec + 'static,
    F: Fn(&C, C::Item) -> Option<C::Item> + Copy + 'static,
    C::Item: Copy + Ord + TryFrom<usize> + Debug,
    usize: TryFrom<C::Item>,
{
    let mut benchmarks = vec![];
    for size in SIZES {
        let name = format!("{}/{}/nodup", type_name::<C::Item>(), size);
        benchmarks.push(benchmark_fn(name, move |b| {
            let mut rnd = RandomCollection::<C>::new(size, 1, b.seed);
            let input = rnd.random_collection();
            b.iter(move || f(&input.collection, rnd.next_needle(&input)))
        }));
    }
    benchmarks
}
