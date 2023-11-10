#!/usr/bin/env bash

# This script is running ordsearch criterion benchmarks and producing log in target folder
# Should be execute in the ordsearch directory

set -eo pipefail

FILE=./target/criterion.txt

if [ -f "${FILE}" ]; then
    rm -f "${FILE}"
fi

for i in {1..30}; do
    cargo +nightly bench \
      --bench=search_comparison \
      --features=nightly \
      "Search u8/(sorted_vec|ordsearch)/8$" >> "${FILE}"
done

for NAME in "u8/sorted_vec/8" "u8/ordsearch/8"; do
    echo "${NAME}"
    cat "${FILE}" | grep "${NAME}" | grep 'time:' | awk '{print $6}'
done


