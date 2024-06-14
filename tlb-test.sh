#!/usr/bin/env bash

set -eo pipefail

cargo export target/benchmarks -t no_prefetch -- bench --bench='search-ord'
cargo export target/benchmarks -t prefetch -- bench --bench='search-ord' --features=prefetch

sudo ./hw_perf/hw_perf -- \
    target/benchmarks/search_ord-prefetch compare target/benchmarks/search_ord-no_prefetch \
    -f 'u*/65536/*' -p -t 1 \
    $@
