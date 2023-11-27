#!/usr/bin/env bash

set -eo pipefail

cargo export target/benchmarks -- bench --bench='search-*'

echo "OrderedCollection vs Vec"
target/benchmarks/search_vec compare target/benchmarks/search_ord $@

echo "OrderedCollection vs BTree"
target/benchmarks/search_btree compare target/benchmarks/search_ord $@
