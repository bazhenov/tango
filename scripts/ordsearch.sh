#!/usr/bin/env bash
set -eo pipefail

cargo +nightly export target/benchmarks -- bench --bench="search-*"

pushd ../ordsearch/
cargo +nightly export ../target/benchmarks -- bench --bench=search_comparison --features=nightly
popd

rm -f target/tango.txt
rm -f target/criterion.txt

for i in {1..1000}; do
    # Tango benchmarks
    (
    target/benchmarks/search_ord compare target/benchmarks/search_vec -f "*/u32/1024/nodup" -t 1 \
        | awk -v OFS=';' -v FS=" {2,}" '{print "tango/u32/1024/1s", $NF}' | tr -d '%*'
    target/benchmarks/search_ord compare target/benchmarks/search_vec -f "*/u32/1024/nodup" -t 0.5 \
        | awk -v OFS=';' -v FS=" {2,}" '{print "tango/u32/1024/0.5s", $NF}' | tr -d '%*'
    target/benchmarks/search_ord compare target/benchmarks/search_vec -f "*/u32/1024/nodup" -t 0.3 \
        | awk -v OFS=';' -v FS=" {2,}" '{print "tango/u32/1024/0.3s", $NF}' | tr -d '%*'
    target/benchmarks/search_ord compare target/benchmarks/search_vec -f "*/u32/1024/nodup" -t 0.1 \
        | awk -v OFS=';' -v FS=" {2,}" '{print "tango/u32/1024/0.1s", $NF}' | tr -d '%*'
    ) | tee -a target/tango.txt

    # Criterion benchmarks
    target/benchmarks/search_comparison "Search u32/(ordsearch|sorted_vec)/1024" \
        | tee -a target/criterion.txt
done

# Reporting code
# paste \
#     <(cat criterion-u32-2.txt | grep -A1 'sorted_vec' | grep 'time:' | awk '{print $4}') \
#     <(cat criterion-u32-2.txt | grep -A1 'ordsearch' | grep 'time:' | awk '{print $4}') \
#     | awk 'OFS=";" {print "criterion/u32/1024", ($2 - $1) / $1 * 100}'
