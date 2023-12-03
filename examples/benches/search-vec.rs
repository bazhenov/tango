#![cfg_attr(feature = "align", feature(fn_align))]

use common::{search_benchmarks, Sample};
use std::process::ExitCode;
use tango_bench::benchmarks;

mod common;

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_vec<T: Copy + Ord>(haystack: &Sample<Vec<T>>, needle: &T) -> Option<T> {
    let haystack = haystack.as_ref();
    haystack
        .binary_search(needle)
        .ok()
        .and_then(|idx| haystack.get(idx))
        .copied()
}

benchmarks!(
    search_benchmarks(search_vec::<u8>),
    search_benchmarks(search_vec::<u16>),
    search_benchmarks(search_vec::<u32>),
    search_benchmarks(search_vec::<u64>)
);

pub fn main() -> tango_bench::cli::Result<ExitCode> {
    common::main()
}
