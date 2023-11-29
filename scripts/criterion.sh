#!/usr/bin/env bash
set -eo pipefail

FILE=./target/criterion.txt

if [ -f "${FILE}" ]; then
    rm -f "${FILE}"
fi

cargo export ./target/benchmarks -- bench --bench=criterion

time (
    for i in {1..30}; do
        ./target/benchmarks/criterion --bench str_length_495 \
            --warm-up-time 1 --measurement-time 1 >> "${FILE}"
        ./target/benchmarks/criterion --bench str_length_500 \
            --warm-up-time 1 --measurement-time 1 >> "${FILE}"
    done
)

paste \
    <(cat "${FILE}" | grep -A1 "str_length_500" | grep 'time:' | awk '{print $5}') \
    <(cat "${FILE}" | grep -A1 "str_length_495" | grep 'time:' | awk '{print $5}') | \
    awk '{print ($2 - $1) / $1 * 100}'
