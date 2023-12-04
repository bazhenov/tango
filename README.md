# Tango.rs

<div align="center">
  <a href="https://crates.io/crates/tango-bench"><img src="https://img.shields.io/crates/v/tango-bench" alt="Tango Bench"/></a>
  <a href="https://docs.rs/tango-bench/latest/tango_bench/"><img src="https://img.shields.io/docsrs/tango-bench" alt="Tango Bench"/></a>
</div>

It used to be that benchmarking required a significant amount of time and numerous iterations to arrive at meaningful results, which was particularly arduous when trying to detect subtle changes, such as those within the range of a few percentage points.

Introducing Tango.rs, a novel benchmarking framework that employs paired benchmarking to assess code performance. This approach capitalizes on the fact that it's far more efficient to measure the performance difference between two simultaneously executing functions compared to two functions executed consecutively.

Features:

- very high sensitivity to changes which allows to converge on results quicker than traditional (pointwise) approach. Often the fraction of a second is enough;
- ability to compare different versions of the same code from different VCS commits (A/B-benchmarking);

## 1 second, 1 percent, 1 error

Compared to traditional pointwise benchmarking, paired benchmarking is significantly more sensitive to changes. This heightened sensitivity enables the early detection of statistically significant performance variations.

Tango is designed to have the capability to detect a 1% change in performance within just 1 second in at least 9 out of 10 test runs.

## Prerequirements

1. Rust and Cargo toolchain installed
2. [`cargo-export`](https://github.com/bazhenov/cargo-export) installed

## Getting started

1. Add cargo dependency and create new benchmark:

   ```toml
   [dev-dependencies]
   tango-bench = "^0.2"

   [[bench]]
   name = "factorial"
   harness = false
   ```

1. Add build script (`build.rs`) which allows benchmarks to export symbols for dynamic linking

   ```rust,ignore
   fn main() {
       println!("cargo:rustc-link-arg-benches=-rdynamic");
   }
   ```

1. Add `benches/factorial.rs` with the following content:

   ```rust,no_run
   use std::hint::black_box;
   use tango_bench::{benchmark_fn, tango_benchmarks, tango_main, IntoBenchmarks};

   pub fn factorial(mut n: usize) -> usize {
       let mut result = 1usize;
       while n > 0 {
           result = result.wrapping_mul(black_box(n));
           n -= 1;
       }
       result
   }

   fn factorial_benchmarks() -> impl IntoBenchmarks {
       [
           benchmark_fn("factorial", || factorial(500)),
       ]
   }

   tango_benchmarks!(factorial_benchmarks());
   tango_main!();
   ```

1. Build and export benchmark to `target/benchmarks` directory:

   ```console
   $ cargo export target/benchmarks -- bench --bench=factorial
   ```

1. Now lets try to modify `factorial.rs` and make factorial faster :)

   ```rust,ignore
   fn factorial_benchmarks() -> impl IntoBenchmarks {
       [
           benchmark_fn("factorial", || factorial(495)),
       ]
   }
   ```

1. Now we can compare new version with already built one:

   ```console
   $ cargo bench -q --bench=factorial -- compare target/benchmarks/factorial
   factorial             [ 375.5 ns ... 369.0 ns ]      -1.58%*
   ```

The result shows that indeed there is indeed ~1% difference between `factorial(500)` and `factorial(495)`.

Additional examples are available in `examples` directory.

## Contributing

The project is in its early stages so any help will be appreciated. Here are some ideas you might find interesting

- find a way to provide a more user friendly API for registering functions in the system
- if you're a library author, trying out tango and providing feedback will be very useful
