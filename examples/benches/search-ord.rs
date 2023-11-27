#![cfg_attr(feature = "align", feature(fn_align))]

use common::search_benchmarks;
use ordsearch::OrderedCollection;
use tango_bench::benchmarks;

mod common;

#[cfg_attr(feature = "align", repr(align(32)))]
#[cfg_attr(feature = "align", inline(never))]
fn search_ord<T: Copy + Ord>(haystack: &impl AsRef<OrderedCollection<T>>, needle: &T) -> Option<T> {
    haystack.as_ref().find_gte(*needle).copied()
}

benchmarks!(search_benchmarks::<u64, _>(search_ord));

pub fn main() {
    common::main()
}
