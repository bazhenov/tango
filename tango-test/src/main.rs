use tango_bench::{benchmark_fn, tango_benchmarks, tango_main};

fn factorial(mut input: u64) -> u64 {
    let mut product = 1u64;
    while input > 1 {
        product = product.wrapping_mul(input);
        input -= 1;
    }
    product
}

tango_benchmarks!([benchmark_fn("factorial", |b| b.iter(|| factorial(500)))]);
tango_main!();
