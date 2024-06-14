#!/usr/bin/env bash
set -eo pipefail

cargo +nightly export target/benches -- bench --bench='search-ord'

pushd target/benches
cp search_ord search_ord2

if [[ "$(uname)" == "Darwin" ]]; then
    codesign --force --deep --sign - search_ord*
fi

rm -rf ../data/*.{csv,svg}
./search_ord compare -d ../data search_ord2 $@
popd
