#!/usr/bin/env bash
set -eo pipefail

cargo +nightly export ./target/benchmarks -- bench --bench='search-*'

mkdir -p target/dump
rm -f target/dump/*.csv

for i in {1..100}; do
    ./target/benchmarks/search_ord compare ./target/benchmarks/search_vec -t 1 -o \
        -f 'search/u32/1024/nodup' -d target/dump --sampler=linear
    mv "target/dump/search-u32-1024-nodup.csv" "target/dump/sample-$i.csv"
done
