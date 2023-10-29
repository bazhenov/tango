#![cfg_attr(feature = "align", feature(fn_align))]

extern crate tango_bench;

use ordsearch::OrderedCollection;
use std::{
    any::type_name,
    collections::BTreeSet,
    convert::TryFrom,
    fmt,
    iter::FromIterator,
    marker::PhantomData,
    ops::Bound,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};
use tango_bench::{benchmark_fn, cli, Benchmark, Generator, MeasurementSettings};

struct RandomVec<T> {
    size: usize,
    max_value: usize,
    _type: PhantomData<T>,
}

impl<T> RandomVec<T>
where
    T: Ord + Copy + TryFrom<usize>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
{
    fn new(size: usize, max_value: usize) -> Self {
        Self {
            size,
            max_value,
            _type: PhantomData,
        }
    }
}

impl<T> Generator for RandomVec<T>
where
    T: Ord + Copy + TryFrom<usize>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
{
    type Haystack = (Rc<Vec<T>>, Rc<OrderedCollection<T>>, Rc<BTreeSet<T>>);
    type Needle = T;

    fn next_haystack(&mut self) -> Self::Haystack {
        let (vec, ord, btree) = generate_new_pair(self.size, self.max_value);
        (Rc::new(vec), Rc::new(ord), Rc::new(btree))
    }

    fn name(&self) -> String {
        format!("Size<{}, {}>", type_name::<T>(), self.size)
    }

    fn next_needle(&mut self) -> Self::Needle {
        pseudorandom_iter(self.max_value).next().unwrap()
    }
}

fn generate_new_pair<T>(
    size: usize,
    max_value: usize,
) -> (Vec<T>, OrderedCollection<T>, BTreeSet<T>)
where
    T: Ord + Copy + TryFrom<usize>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
{
    let mut vec = Vec::with_capacity(size);
    let mut rand = pseudorandom_iter(max_value);
    for _ in 0..size {
        vec.push(rand.next().unwrap());
    }
    vec.sort();
    let ord = OrderedCollection::from_sorted_iter(vec.iter().copied());
    let btree = BTreeSet::from_iter(vec.iter().copied());
    (vec, ord, btree)
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
    max_value: usize,
) -> Benchmark<(Rc<Vec<T>>, Rc<OrderedCollection<T>>, Rc<BTreeSet<T>>), T, Option<T>>
where
    T: Copy + Ord + TryFrom<usize> + 'static,
    <T as TryFrom<usize>>::Error: fmt::Debug,
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

    let generators = sizes
        .into_iter()
        .map(|size| RandomVec::<T>::new(size, max_value));
    b.add_generators(generators);

    b
}

fn main() {
    let settings = MeasurementSettings {
        samples_per_haystack: 1_000_000,
        max_iterations_per_sample: 1,
        ..Default::default()
    };

    cli::run(create_benchmark::<u8>(u8::max_value() as usize), settings);
    cli::run(create_benchmark::<u16>(u16::max_value() as usize), settings);
    cli::run(create_benchmark::<u32>(u32::max_value() as usize), settings);
    cli::run(create_benchmark::<u64>(u64::max_value() as usize), settings);
    cli::run(
        create_benchmark::<u128>(u128::max_value() as usize),
        settings,
    );
}

fn pseudorandom_iter<T>(max: usize) -> impl Iterator<Item = T>
where
    T: TryFrom<usize>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
{
    static SEED: AtomicUsize = AtomicUsize::new(0);
    let mut seed = SEED.fetch_add(1, Ordering::SeqCst);

    std::iter::from_fn(move || {
        // LCG constants from https://en.wikipedia.org/wiki/Numerical_Recipes.
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        SEED.store(seed, Ordering::SeqCst);

        let r = seed % max;
        Some(T::try_from(r).unwrap())
    })
}
