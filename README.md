# Rust Paired Testing Prototype

Proof-of-concept of paired tests idea described in the article [Paired benchmarking. How to measure performance](https://www.bazhenov.me/posts/paired-benchmarking/)

Running paired tests:

```console
$ cargo run --release
name                                          B min      C min      min ∆     B mean     C mean     mean ∆ mean ∆ (%)
std / std                                       731        731       0.0%      868.5      868.9        0.1       0.0%
std_count / std_count                          4210       4211       0.0%     6942.5     6926.8        1.7       0.0%
std_count_rev / std_count_rev                  4211       4211       0.0%     7160.3     7155.0        1.6       0.0%
std_5000 / std_4925                            3167       3121      -1.5%     4573.9     4507.7      -65.9      -1.4% CHANGE DETECTED
std_count / std_count_rev                      4210       4210       0.0%     6952.8     7205.3      236.5       3.4% CHANGE DETECTED
std / std_count                                 740       4210     468.9%      890.7     6952.0     6062.8     680.7% CHANGE DETECTED
```

Running pointwise tests using criterion.rs:

```console
$ cargo bench
utf8/std_count          time:   [9.6316 µs 9.6550 µs 9.6798 µs]
Found 6 outliers among 100 measurements (6.00%)
  3 (3.00%) high mild
  3 (3.00%) high severe
utf8/std_count_rev      time:   [7.5723 µs 7.7677 µs 7.9667 µs]
Found 8 outliers among 100 measurements (8.00%)
  8 (8.00%) high severe
```
