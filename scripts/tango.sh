#!/usr/bin/env bash
set -eo pipefail

FILE=./target/tango.txt

if [ -f "${FILE}" ]; then
    rm -f "${FILE}"
fi

cargo export target/benchmarks -- bench --bench='tango-*'

time (
    for i in {1..30}; do
        ./target/benchmarks/tango_faster compare ./target/benchmarks/tango_slower \
            -t 1000 -o -f 'str_length_limit' >> "${FILE}"
    done
)

cat "${FILE}" | awk '{print $(NF)}' | sed 's/%//'
