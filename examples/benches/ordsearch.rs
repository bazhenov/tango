#![cfg_attr(feature = "align", feature(fn_align))]

extern crate tango_bench;

use num_traits::{bounds::UpperBounded, ToPrimitive};
use ordsearch::OrderedCollection;
use std::{
    any::type_name, collections::BTreeSet, convert::TryFrom, iter::FromIterator,
    marker::PhantomData, ops::Bound, usize,
};
use tango_bench::{benchmark_fn, cli, Benchmark, Generator, MeasurementSettings};

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

struct RandomVec<T> {
    rng: Lcg<T>,
    size: usize,
}

impl<T> RandomVec<T>
where
    T: Ord + Copy + TryFrom<usize>,
{
    fn new(size: usize) -> Self {
        Self {
            rng: Lcg::new(0),
            size,
        }
    }
}

struct Sample<T> {
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

    fn name(&self) -> String {
        format!("Size<{}, {}>", type_name::<T>(), self.size)
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

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_ord<T: Copy + Ord>(haystack: &impl AsRef<OrderedCollection<T>>, needle: &T) -> Option<T> {
    haystack.as_ref().find_gte(*needle).copied()
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_btree<T: Copy + Ord>(haystack: &impl AsRef<BTreeSet<T>>, needle: &T) -> Option<T> {
    haystack
        .as_ref()
        .range((Bound::Included(needle), Bound::Unbounded))
        .next()
        .copied()
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_vec<T: Copy + Ord>(haystack: &impl AsRef<Vec<T>>, needle: &T) -> Option<T> {
    let haystack = haystack.as_ref();
    haystack
        .binary_search(needle)
        .ok()
        .and_then(|idx| haystack.get(idx))
        .copied()
}

fn create_benchmark<T>() -> Benchmark<Sample<T>, T>
where
    T: Copy + Ord + TryFrom<usize> + UpperBounded + ToPrimitive + 'static,
    usize: TryFrom<T>,
{
    let mut b = Benchmark::default();

    b.add_pair(
        benchmark_fn("vec", search_vec),
        benchmark_fn("ord", search_ord),
    );
    b.add_pair(
        benchmark_fn("btree", search_btree),
        benchmark_fn("ord", search_ord),
    );

    let sizes = [
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4069, 8192, 16384, 32768, 65536,
    ];

    let max_value = T::max_value().to_usize().unwrap_or(usize::max_value());
    let generators = sizes
        .into_iter()
        .filter(|s| *s <= max_value)
        .map(RandomVec::<T>::new);
    b.add_generators(generators);

    b
}

fn main() {
    let settings = MeasurementSettings {
        samples_per_haystack: 1_000,
        ..Default::default()
    };

    cli::run(create_benchmark::<u8>(), settings);
    cli::run(create_benchmark::<u16>(), settings);
    cli::run(create_benchmark::<u32>(), settings);
    cli::run(create_benchmark::<u64>(), settings);
    cli::run(create_benchmark::<u128>(), settings);
}
