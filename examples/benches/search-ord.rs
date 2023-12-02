#![cfg_attr(feature = "align", feature(fn_align))]

use common::{FromSortedVec, RandomCollection, SearchBenchmarks};
use ordsearch::OrderedCollection;
use std::process::ExitCode;
use tango_bench::benchmarks;

mod common;

type RandomOrderedCollection<T> = RandomCollection<OrderedCollection<T>, T>;

impl<T: Ord> FromSortedVec<T> for OrderedCollection<T> {
    fn from_sorted_vec(v: Vec<T>) -> Self {
        OrderedCollection::from_sorted_iter(v)
    }
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_ord<T: Copy + Ord>(haystack: &impl AsRef<OrderedCollection<T>>, needle: &T) -> Option<T> {
    haystack.as_ref().find_gte(*needle).copied()
}

benchmarks!(
    SearchBenchmarks(RandomOrderedCollection::<u8>::new, search_ord),
    SearchBenchmarks(RandomOrderedCollection::<u16>::new, search_ord),
    SearchBenchmarks(RandomOrderedCollection::<u32>::new, search_ord),
    SearchBenchmarks(RandomOrderedCollection::<u64>::new, search_ord)
);

pub fn main() -> tango_bench::cli::Result<ExitCode> {
    common::main()
}
