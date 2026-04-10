use tango_bench::{benchmark_fn, tango_benchmarks};

tango_benchmarks!([benchmark_fn("panicking", |_| panic!("Ooops"))]);
