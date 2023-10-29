#![cfg_attr(feature = "align", feature(fn_align))]

extern crate rust_pairwise_testing;

use ordsearch::OrderedCollection;
use rust_pairwise_testing::{benchmark_fn, cli, Benchmark, Generator, MeasurementSettings};
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
    time::Duration,
};

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

fn main() {
    let mut b = Benchmark::default();

    b.add_pair(
        benchmark_fn("vec", search_vec),
        benchmark_fn("ord", search_ord),
    );
    b.add_pair(
        benchmark_fn("btree", search_btree),
        benchmark_fn("ord", search_ord),
    );

    let mut payloads = (10..20)
        .into_iter()
        .map(|i| 2usize.pow(i))
        .map(|size| RandomVec::<u32>::new(size, u32::max_value() as usize))
        .collect::<Vec<_>>();

    let mut refs = payloads.iter_mut().map(|i| i as _).collect::<Vec<_>>();

    let settings = MeasurementSettings {
        max_samples: 1_000_000,
        max_duration: Duration::from_millis(100),
        outlier_detection_enabled: true,
        samples_per_haystack: 1_000_000,
        samples_per_needle: 1,
        max_iterations_per_sample: 1,
    };

    cli::run(b, settings, &mut refs[..]);
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
