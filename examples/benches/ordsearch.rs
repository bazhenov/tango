#![cfg_attr(feature = "align", feature(fn_align))]

extern crate tango_bench;

use num_traits::{bounds::UpperBounded, ToPrimitive};
use ordsearch::OrderedCollection;
use std::{
    any::type_name, collections::BTreeSet, convert::TryFrom, iter::FromIterator,
    marker::PhantomData, ops::Bound, rc::Rc, usize,
};
use tango_bench::{benchmark_fn, cli, Benchmark, Generator, MeasurementSettings};

struct LCG<T> {
    value: usize,
    _type: PhantomData<T>,
}

impl<T> LCG<T>
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
    rng: LCG<T>,
    size: usize,
    last_max: usize,
}

impl<T> RandomVec<T>
where
    T: Ord + Copy + TryFrom<usize>,
{
    fn new(size: usize) -> Self {
        Self {
            rng: LCG::new(0),
            size,
            last_max: 0,
        }
    }

    fn generate_new(&mut self, size: usize) -> (Vec<T>, OrderedCollection<T>, BTreeSet<T>)
    where
        T: Ord + Copy + TryFrom<usize>,
    {
        let mut vec = (0..size)
            .map(|v| 2 * v)
            .map(|v| T::try_from(v))
            .take_while(|r| r.is_ok())
            .collect::<Result<Vec<_>, _>>()
            .ok()
            .unwrap();
        vec.sort();
        let ord = OrderedCollection::from_sorted_iter(vec.iter().copied());
        let btree = BTreeSet::from_iter(vec.iter().copied());
        (vec, ord, btree)
    }
}

impl<T> Generator for RandomVec<T>
where
    T: Ord + Copy + TryFrom<usize>,
    usize: TryFrom<T>,
{
    type Haystack = (Rc<Vec<T>>, Rc<OrderedCollection<T>>, Rc<BTreeSet<T>>);
    type Needle = T;

    fn next_haystack(&mut self) -> Self::Haystack {
        let (vec, ord, btree) = self.generate_new(self.size);
        self.last_max = usize::try_from(vec.iter().copied().max().unwrap())
            .ok()
            .unwrap();

        (Rc::new(vec), Rc::new(ord), Rc::new(btree))
    }

    fn name(&self) -> String {
        format!("Size<{}, {}>", type_name::<T>(), self.size)
    }

    fn next_needle(&mut self) -> Self::Needle {
        if self.last_max > 0 {
            let next = self.rng.next(self.last_max + 1);
            next
        } else {
            self.rng.next(self.size * 2 + 1)
        }
    }

    fn reset(&mut self) {
        self.rng = LCG::new(0);
    }
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_ord<T: Copy + Ord>(
    haystack: &(
        impl AsRef<Vec<T>>,
        impl AsRef<OrderedCollection<T>>,
        impl AsRef<BTreeSet<T>>,
    ),
    needle: &T,
) -> Option<T> {
    let (_, collection, _) = haystack;
    collection.as_ref().find_gte(*needle).copied()
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_btree<T: Copy + Ord>(
    haystack: &(
        impl AsRef<Vec<T>>,
        impl AsRef<OrderedCollection<T>>,
        impl AsRef<BTreeSet<T>>,
    ),
    needle: &T,
) -> Option<T> {
    let (_, _, collection) = haystack;
    collection
        .as_ref()
        .range((Bound::Included(needle), Bound::Unbounded))
        .next()
        .map(|v| *v)
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_vec<T: Copy + Ord>(
    haystack: &(
        impl AsRef<Vec<T>>,
        impl AsRef<OrderedCollection<T>>,
        impl AsRef<BTreeSet<T>>,
    ),
    needle: &T,
) -> Option<T> {
    let (collection, _, _) = haystack;
    collection
        .as_ref()
        .binary_search(needle)
        .ok()
        .and_then(|idx| collection.as_ref().get(idx))
        .copied()
}

fn create_benchmark<T>(
) -> Benchmark<(Rc<Vec<T>>, Rc<OrderedCollection<T>>, Rc<BTreeSet<T>>), T, Option<T>>
where
    T: Copy + Ord + TryFrom<usize> + 'static + UpperBounded + ToPrimitive,
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
        samples_per_haystack: 1_000_000,
        ..Default::default()
    };

    cli::run(create_benchmark::<u8>(), settings);
    cli::run(create_benchmark::<u16>(), settings);
    cli::run(create_benchmark::<u32>(), settings);
    cli::run(create_benchmark::<u64>(), settings);
    cli::run(create_benchmark::<u128>(), settings);
}
