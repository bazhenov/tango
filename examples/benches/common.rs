#![cfg_attr(feature = "align", feature(fn_align))]

extern crate tango_bench;

use std::{any::type_name, convert::TryFrom, fmt::Debug, iter, marker::PhantomData};
use tango_bench::{
    BenchmarkMatrix, Generator, IntoBenchmarks, MeasurementSettings, DEFAULT_SETTINGS,
};

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
    name: String,
    value_dup_factor: usize,
    phantom: PhantomData<C>,
}

impl<C: FromSortedVec> RandomCollection<C>
where
    C::Item: Ord + Copy + TryFrom<usize>,
{
    pub fn new(size: usize, value_dup_factor: usize) -> Self {
        let type_name = type_name::<C::Item>();
        let name = if value_dup_factor > 1 {
            format!("{}/{}/dup-{}", type_name, size, value_dup_factor)
        } else {
            format!("{}/{}/nodup", type_name, size)
        };

        Self {
            rng: Lcg(0),
            size,
            value_dup_factor,
            name,
            phantom: PhantomData,
        }
    }
}

impl<C: FromSortedVec> Generator for RandomCollection<C>
where
    C::Item: Ord + Copy + TryFrom<usize> + Debug,
    usize: TryFrom<C::Item>,
{
    type Haystack = Sample<C>;
    type Needle = C::Item;

    fn next_haystack(&mut self) -> Self::Haystack {
        let vec = generate_sorted_vec(self.size, self.value_dup_factor);
        let max = usize::try_from(*vec.last().unwrap()).ok().unwrap();

        Sample {
            collection: C::from_sorted_vec(vec),
            max_value: max,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn next_needle(&mut self, sample: &Self::Haystack) -> Self::Needle {
        self.rng.next(sample.max_value + 1)
    }

    fn sync(&mut self, seed: u64) {
        self.rng = Lcg(seed as usize);
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
        .flat_map(|v| iter::repeat(v).take(dup_factor))
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
    F: Fn(&Sample<C>, &C::Item) -> Option<C::Item> + Copy + 'static,
    C::Item: Copy + Ord + TryFrom<usize> + Debug,
    usize: TryFrom<C::Item>,
{
    BenchmarkMatrix::with_params(SIZES, |size| RandomCollection::<C>::new(size, 1))
        .add_generators_with_params(SIZES, |size| RandomCollection::<C>::new(size, 16))
        .add_function("search", f)
        .into_benchmarks()
}

pub const SETTINGS: MeasurementSettings = MeasurementSettings {
    samples_per_haystack: usize::max_value(),
    max_iterations_per_sample: 10_000,
    ..DEFAULT_SETTINGS
};
