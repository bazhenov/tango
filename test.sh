#!/usr/bin/env bash
set -eo pipefail

cargo +nightly export target/benches -- bench --bench='search-ord'

pushd target/benches
cp search_ord search_ord2

if [[ "$(uname)" == "Darwin" ]]; then
    codesign --force --deep --sign - search_ord*
fi

rm -rf ../data/*.csv
./search_ord compare -d ../data search_ord2 $@
popd

if [[ -x "$(command -v gnuplot)" ]]; then
    for csv_file in target/data/*.csv; do
        if [[ -f "$csv_file" ]]; then
            gnuplot -c pair-test.gnuplot $csv_file "${csv_file%.csv}.svg"
        fi
    done
fi
