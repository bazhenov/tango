#![cfg_attr(feature = "align", feature(fn_align))]

extern crate tango_bench;

use std::{any::type_name, convert::TryFrom, marker::PhantomData, process::ExitCode};
use tango_bench::{cli, BenchmarkMatrix, Generator, IntoBenchmarks, MeasurementSettings};

const SIZES: [usize; 14] = [
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4069, 8192, 16384, 32768, 65536,
];

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
pub struct RandomCollection<C, T> {
    rng: Lcg<T>,
    size: usize,
    name: String,
    phantom: PhantomData<C>,
}

impl<T, C> RandomCollection<C, T>
where
    T: Ord + Copy + TryFrom<usize>,
{
    pub fn new(size: usize) -> Self {
        Self {
            rng: Lcg::new(0),
            size,
            name: format!("{}/{}", type_name::<T>(), size),
            phantom: PhantomData,
        }
    }
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

pub trait FromSortedVec<T> {
    fn from_sorted_vec(v: Vec<T>) -> Self;
}

impl<T> FromSortedVec<T> for Vec<T> {
    fn from_sorted_vec(v: Vec<T>) -> Self {
        v
    }
}

impl<T, C: FromSortedVec<T>> Generator for RandomCollection<C, T>
where
    T: Ord + Copy + TryFrom<usize>,
    usize: TryFrom<T>,
{
    type Haystack = Sample<C>;
    type Needle = T;

    fn next_haystack(&mut self) -> Self::Haystack {
        let vec = generate_sorted_vec(self.size);
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

pub struct SearchBenchmarks<SearchF, GeneratorF>(pub GeneratorF, pub SearchF);

impl<F, R, G, H, O, N> IntoBenchmarks for SearchBenchmarks<F, R>
where
    F: Fn(&H, &N) -> O + Copy + 'static,
    R: Fn(usize) -> G,
    H: 'static,
    G: Generator<Haystack = H, Needle = N> + 'static,
{
    fn into_benchmarks(self) -> Vec<Box<dyn tango_bench::MeasureTarget>> {
        let SearchBenchmarks(generator_func, search_func) = self;

        BenchmarkMatrix::with_params(SIZES, generator_func)
            .add_function("search", search_func)
            .into_benchmarks()
    }
}

pub fn main() -> tango_bench::cli::Result<ExitCode> {
    let settings = MeasurementSettings {
        samples_per_haystack: 1_000,
        ..Default::default()
    };

    cli::run(settings)
}
