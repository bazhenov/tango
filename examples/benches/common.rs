#![cfg_attr(feature = "align", feature(fn_align))]

extern crate tango_bench;

use ordsearch::OrderedCollection;
use std::{
    any::type_name, collections::BTreeSet, convert::TryFrom, iter::FromIterator,
    marker::PhantomData, process::ExitCode,
};
use tango_bench::{cli, BenchmarkMatrix, Generator, IntoBenchmarks, MeasurementSettings};

#[derive(Clone)]
struct Lcg<T> {
    value: usize,
    _type: PhantomData<T>,
}

impl<T> Lcg<T>
where
    T: TryFrom<usize>,
{
    fn new(seed: usize) -> Self {
        Self {
            value: seed,
            _type: PhantomData,
        }
    }

    fn next(&mut self, max_value: usize) -> T {
        self.value = self.value.wrapping_mul(1664525).wrapping_add(1013904223);
        T::try_from((self.value >> 32) % max_value).ok().unwrap()
    }
}

#[derive(Clone)]
struct RandomVec<T> {
    rng: Lcg<T>,
    size: usize,
    name: String,
}

impl<T> RandomVec<T>
where
    T: Ord + Copy + TryFrom<usize>,
{
    fn new(size: usize) -> Self {
        Self {
            rng: Lcg::new(0),
            size,
            name: format!("Vec<{}, {}>", type_name::<T>(), size),
        }
    }
}

pub struct Sample<T> {
    vec: Vec<T>,
    ord: OrderedCollection<T>,
    btree: BTreeSet<T>,
}

impl<T> AsRef<BTreeSet<T>> for Sample<T> {
    fn as_ref(&self) -> &BTreeSet<T> {
        &self.btree
    }
}

impl<T> AsRef<OrderedCollection<T>> for Sample<T> {
    fn as_ref(&self) -> &OrderedCollection<T> {
        &self.ord
    }
}

impl<T> AsRef<Vec<T>> for Sample<T> {
    fn as_ref(&self) -> &Vec<T> {
        &self.vec
    }
}

impl<T> Generator for RandomVec<T>
where
    T: Ord + Copy + TryFrom<usize>,
    usize: TryFrom<T>,
{
    type Haystack = Sample<T>;
    type Needle = T;

    fn next_haystack(&mut self) -> Self::Haystack {
        let vec = generate_sorted_vec(self.size);
        let ord = OrderedCollection::from_sorted_iter(vec.iter().copied());
        let btree = BTreeSet::from_iter(vec.iter().copied());

        Sample { vec, ord, btree }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn next_needle(&mut self, haystack: &Self::Haystack) -> Self::Needle {
        let max = (haystack.vec.len() - 1) * 2;
        self.rng.next(max + 1)
    }

    fn reset(&mut self) {
        self.rng = Lcg::new(0);
    }
}

fn generate_sorted_vec<T>(size: usize) -> Vec<T>
where
    T: Ord + Copy + TryFrom<usize>,
{
    (0..size)
        .map(|v| 2 * v)
        .map(|v| T::try_from(v))
        .take_while(|r| r.is_ok())
        .collect::<Result<Vec<_>, _>>()
        .ok()
        .unwrap()
}

pub fn search_benchmarks<T, F>(search_func: F) -> impl IntoBenchmarks
where
    T: Ord + Copy + TryFrom<usize> + 'static,
    usize: TryFrom<T>,
    F: Fn(&Sample<T>, &T) -> Option<T> + Copy + 'static,
{
    let sizes = [
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4069, 8192, 16384, 32768, 65536,
    ];

    BenchmarkMatrix::with_params(sizes, RandomVec::<T>::new).add_function("search", search_func)
}

pub fn main() -> tango_bench::cli::Result<ExitCode> {
    let settings = MeasurementSettings {
        samples_per_haystack: 1_000,
        ..Default::default()
    };

    cli::run(settings)
}
