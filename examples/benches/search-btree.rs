#![cfg_attr(feature = "align", feature(fn_align))]

use common::{FromSortedVec, RandomCollection, SearchBenchmarks};
use std::{collections::BTreeSet, ops::Bound, process::ExitCode};
use tango_bench::benchmarks;

mod common;

type RandomBTreeSet<T> = RandomCollection<BTreeSet<T>, T>;

impl<T: Ord> FromSortedVec<T> for BTreeSet<T> {
    fn from_sorted_vec(v: Vec<T>) -> Self {
        BTreeSet::from_iter(v)
    }
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

benchmarks!(
    SearchBenchmarks(RandomBTreeSet::<u8>::new, search_btree),
    SearchBenchmarks(RandomBTreeSet::<u16>::new, search_btree),
    SearchBenchmarks(RandomBTreeSet::<u32>::new, search_btree),
    SearchBenchmarks(RandomBTreeSet::<u64>::new, search_btree)
);

pub fn main() -> tango_bench::cli::Result<ExitCode> {
    common::main()
}
