#![cfg_attr(feature = "align", feature(fn_align))]

use common::search_benchmarks;
use tango_bench::{tango_benchmarks, tango_main};

mod common;

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_vec<T: Copy + Ord>(haystack: &Vec<T>, needle: T) -> Option<T> {
    haystack
        .binary_search(&needle)
        .ok()
        .and_then(|idx| haystack.get(idx))
        .copied()
}

tango_benchmarks!(
    search_benchmarks(search_vec::<u8>),
    search_benchmarks(search_vec::<u16>),
    search_benchmarks(search_vec::<u32>),
    search_benchmarks(search_vec::<u64>)
);

tango_main!(common::SETTINGS);
