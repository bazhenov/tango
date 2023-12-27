#!/usr/bin/env bash
set -eo pipefail

cargo +nightly export ./target/benchmarks -- bench --bench='tango-*' --features=hw-timer

mkdir -p target/dump
rm -f target/dump/*.csv

TARGET=target/benchmarks/tango_faster

"$TARGET" compare "$TARGET" -d target/dump $@
python3 ./scripts/describe.py target/dump/*.csv
