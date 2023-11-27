#![cfg_attr(feature = "align", feature(fn_align))]

use common::search_benchmarks;
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

benchmarks!(search_benchmarks::<u64, _>(search_vec));

pub fn main() {
    common::main()
}
