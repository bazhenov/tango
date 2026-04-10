use tango_bench::{benchmark_fn, tango_benchmarks};

tango_benchmarks!([benchmark_fn("sleep", |b| b.iter(|| panic!("Intended panic")))]);
