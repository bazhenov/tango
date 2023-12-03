#!/usr/bin/env bash
set -eo pipefail

cargo export target/benchmarks -- bench --bench="search-*"

for i in {1..1000}; do
    target/benchmarks/search_ord compare target/benchmarks/search_vec -f "u32" -t 10 -o \
        | awk -v OFS=';' -v FS=" {2,}" '{print $1, $NF}' | tr -d '%*'
done
