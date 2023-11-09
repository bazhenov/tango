#!/usr/bin/env bash

set -eo pipefail

cargo +nightly export ./benches -- bench --bench=ordsearch

(for i in {1..30}; do
    ./benches/ordsearch pair "<u8, 8>" -o
done) | tee target/ordsearch.txt

paste \
    <(cat target/ordsearch.txt | grep "1_" | awk '{print ($(NF)) + 0.0}') \
    <(cat target/ordsearch.txt | grep "2_" | awk '{print ($(NF)) + 0.0}') \
    <(cat target/ordsearch.txt | grep "3_" | awk '{print ($(NF)) + 0.0}')
