#![cfg_attr(feature = "align", feature(fn_align))]

extern crate tango_bench;

use ordsearch::OrderedCollection;
use std::{
    any::type_name,
    collections::BTreeSet,
    convert::TryFrom,
    fmt::{self, Debug},
    iter::{self, FromIterator},
    marker::PhantomData,
    ops::Bound,
    rc::Rc,
};
use tango_bench::{benchmark_fn, cli, Benchmark, Generator, MeasurementSettings};

struct RandomVec<T> {
    seed: usize,
    size: usize,
    max_value: usize,
    _type: PhantomData<T>,
}

impl<T: Debug> RandomVec<T>
where
    T: Ord + Copy + TryFrom<usize>,
{
    fn new(size: usize, max_value: usize) -> Self {
        Self {
            seed: 0,
            size,
            max_value,
            _type: PhantomData,
        }
    }

    fn generate_new(
        &mut self,
        size: usize,
        max_value: usize,
    ) -> (Vec<T>, OrderedCollection<T>, BTreeSet<T>)
    where
        T: Ord + Copy + TryFrom<usize>,
    {
        let mut vec = self.random(max_value, 2).take(size).collect::<Vec<_>>();
        vec.sort();
        // println!("{:?}", vec);
        let ord = OrderedCollection::from_sorted_iter(vec.iter().copied());
        let btree = BTreeSet::from_iter(vec.iter().copied());
        (vec, ord, btree)
    }

    fn random(&mut self, max: usize, factor: usize) -> impl Iterator<Item = T>
    where
        T: TryFrom<usize>,
    {
        let mut value = self.seed;
        self.seed += 1;

        iter::from_fn(move || {
            // LCG constants from https://en.wikipedia.org/wiki/Numerical_Recipes.
            value = value.wrapping_mul(1664525).wrapping_add(1013904223);
            Some(T::try_from(((value >> 32) * factor) % max).ok().unwrap())
        })
    }
}

impl<T> Generator for RandomVec<T>
where
    T: Ord + Copy + TryFrom<usize> + Debug,
    <T as TryFrom<usize>>::Error: fmt::Debug,
{
    type Haystack = (Rc<Vec<T>>, Rc<OrderedCollection<T>>, Rc<BTreeSet<T>>);
    type Needle = T;

    fn next_haystack(&mut self) -> Self::Haystack {
        let (vec, ord, btree) = self.generate_new(self.size, self.max_value);
        (Rc::new(vec), Rc::new(ord), Rc::new(btree))
    }

    fn name(&self) -> String {
        format!("Size<{}, {}>", type_name::<T>(), self.size)
    }

    fn next_needle(&mut self) -> Self::Needle {
        self.random(self.max_value, 1).next().unwrap()
    }

    fn reset(&mut self) {
        self.seed = 0;
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

fn create_benchmark<T: Debug>(
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
        samples_per_haystack: 1_000,
        // max_iterations_per_sample: 100_000,
        // min_iterations_per_sample: 1000,
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
