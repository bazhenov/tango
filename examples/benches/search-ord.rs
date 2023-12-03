#![cfg_attr(feature = "align", feature(fn_align))]

use common::{search_benchmarks, FromSortedVec};
use ordsearch::OrderedCollection;
use std::process::ExitCode;
use tango_bench::benchmarks;

mod common;

impl<T: Ord> FromSortedVec for OrderedCollection<T> {
    type Item = T;
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
    search_benchmarks(search_ord::<u8>),
    search_benchmarks(search_ord::<u16>),
    search_benchmarks(search_ord::<u32>),
    search_benchmarks(search_ord::<u64>)
);

pub fn main() -> tango_bench::cli::Result<ExitCode> {
    common::main()
}
