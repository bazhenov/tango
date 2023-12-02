#![cfg_attr(feature = "align", feature(fn_align))]

use common::search_benchmarks;
use std::{collections::BTreeSet, ops::Bound, process::ExitCode};
use tango_bench::benchmarks;

mod common;

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
    search_benchmarks::<u8, _>(search_btree),
    search_benchmarks::<u16, _>(search_btree),
    search_benchmarks::<u32, _>(search_btree),
    search_benchmarks::<u64, _>(search_btree)
);

pub fn main() -> tango_bench::cli::Result<ExitCode> {
    common::main()
}
