#!/usr/bin/env bash
set -eo pipefail

cargo +nightly export ./target/benchmarks -- bench --bench='search-*'

mkdir -p target/dump
rm -f target/dump/*.csv

for i in {1..30}; do
    for sampler in flat linear random; do
        printf "%10s : " "$sampler"
        ./target/benchmarks/search_ord compare ./target/benchmarks/search_vec -t 1 \
            -f 'search/u32/1024/nodup' -d target/dump --sampler="$sampler"
        mv "target/dump/search-u32-1024-nodup.csv" "target/dump/$sampler-$i.csv"
    done
done
