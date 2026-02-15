#![cfg_attr(feature = "align", feature(fn_align))]

use common::{search_benchmarks, FromSortedVec};
use std::{collections::BTreeSet, ops::Bound};
use tango_bench::{tango_benchmarks, tango_main};

mod common;

impl<T: Ord> FromSortedVec for BTreeSet<T> {
    type Item = T;

    fn from_sorted_vec(v: Vec<T>) -> Self {
        BTreeSet::from_iter(v)
    }
}

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_btree<T: Copy + Ord>(haystack: &BTreeSet<T>, needle: T) -> Option<T> {
    haystack
        .range((Bound::Included(needle), Bound::Unbounded))
        .next()
        .copied()
}

tango_benchmarks!(
    search_benchmarks(search_btree::<u8>),
    search_benchmarks(search_btree::<u16>),
    search_benchmarks(search_btree::<u32>),
    search_benchmarks(search_btree::<u64>)
);
tango_main!();
