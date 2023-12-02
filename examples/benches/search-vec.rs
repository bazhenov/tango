#![cfg_attr(feature = "align", feature(fn_align))]

use common::{RandomCollection, SearchBenchmarks};
use std::process::ExitCode;
use tango_bench::benchmarks;

mod common;

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_vec<T: Copy + Ord>(haystack: &impl AsRef<Vec<T>>, needle: &T) -> Option<T> {
    let haystack = haystack.as_ref();
    haystack
        .binary_search(needle)
        .ok()
        .and_then(|idx| haystack.get(idx))
        .copied()
}

benchmarks!(
    SearchBenchmarks(RandomCollection::<Vec<u8>, _>::new, search_vec),
    SearchBenchmarks(RandomCollection::<Vec<u16>, _>::new, search_vec),
    SearchBenchmarks(RandomCollection::<Vec<u32>, _>::new, search_vec),
    SearchBenchmarks(RandomCollection::<Vec<u64>, _>::new, search_vec)
);

pub fn main() -> tango_bench::cli::Result<ExitCode> {
    common::main()
}
