# Tango.rs

It used to be that benchmarking requires a lot of time and a lot of iterations to converge on meaningful results. It is espectially painfull when you need to detect small changes – in the order of magnitude of several percents.

Tango.rs is the new benchmarking framework which uses pairwise benchmarking as a way of measuring code performance. It relies on the fact that it's much easier to measure difference in performance of two simultaneusly running functions than of two functions running one after the another.

## 1 second, 1 percent, 1 error

Comparing to classical (pointwise) benchmarking it's much more sensitive to changes which allows to detect statistically significant changes much earlier.

Tango is created to be able to detect 1% change in performance within 1 second in at least 9 runs out of 10.

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