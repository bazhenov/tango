# Tango.rs

It used to be that benchmarking required a significant amount of time and numerous iterations to arrive at meaningful results, which was particularly arduous when trying to detect subtle changes, such as those within the range of a few percentage points.

Introducing Tango.rs, a novel benchmarking framework that employs pairwise benchmarking to assess code performance. This approach capitalizes on the fact that it's far more efficient to measure the performance difference between two simultaneously executing functions compared to two functions executed consecutively.

## 1 second, 1 percent, 1 error

Compared to traditional pointwise benchmarking, pairwise benchmarking is significantly more sensitive to changes. This heightened sensitivity enables the early detection of statistically significant performance variations.

Tango is designed to have the capability to detect a 1% change in performance within just one second in at least 9 out of 10 test runs.

## Getting Started

```toml
[dev-dependencies]
tango-bench = "0.1.*"

[[bench]]
name = "bench"
harness = false
```

Add `benches/bench.rs` with the following content:

```rust
use tango_bench::{benchmark_fn, benchmark_fn_with_setup, cli::run, Benchmark, Generator, StaticValue};

pub fn factorial(mut n: usize) -> usize {
  let mut result = 1usize;
  while n > 0 {
    result = result.wrapping_mul(black_box(n));
    n -= 1;
  }
  result
}

fn main() {
  let mut b = Benchmark::default();
  b.add_generator(StaticValue((), ()));

  benchmark.add_pair(
    benchmark_fn("factorial_500", |_, _| factorial(500)),
    benchmark_fn("factorial_495", |_, _| factorial(495)),
  );

  let settings = MeasurementSettings::default();
  cli::run(b, settings)
}
```

Run benchmarks with following command:

```console
$ cargo run --bench=bench -- pair
```