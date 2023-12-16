#!/usr/bin/env bash
set -eo pipefail

cargo +nightly export target/benchmarks -- bench --bench="search-*"

pushd ../ordsearch/
cargo +nightly export ../tango/target/benchmarks -- bench --bench=search_comparison --features=nightly
popd

rm -f target/tango.txt
rm -f target/criterion.txt

# Patching PIE executable if needed
target/benchmarks/search_ord compare target/benchmarks/search_vec -f "*/u32/32768/nodup" -t 0.1 > /dev/null
if [ -f target/benchmarks/search_vec.patched ]; then
    mv target/benchmarks/search_vec.patched target/benchmarks/search_vec
    chmod +x target/benchmarks/search_vec
fi

for i in {1..1000}; do
    # Tango benchmarks
    (
    for time in 0.1 0.3 0.5 1; do
        target/benchmarks/search_ord compare target/benchmarks/search_vec -f "*/u32/32768/nodup" -t "$time" -o \
            | awk -v OFS=';' -v FS=" {2,}" -v time="$time" '{print "tango/u32/32768/" time "s", $NF}' | tr -d '%*'
    done
    ) | tee -a target/tango.txt

    # Criterion benchmarks
    target/benchmarks/search_comparison --bench "Search u32/(ordsearch|sorted_vec)/1024" \
        | tee -a target/criterion.txt
done

# Reporting code
paste \
    <(cat target/criterion.txt | grep -A1 'sorted_vec' | grep 'time:' | awk '{print $4}') \
    <(cat target/criterion.txt | grep -A1 'ordsearch' | grep 'time:' | awk '{print $4}') \
    | awk 'OFS=";" {print "criterion/u32/1024", ($2 - $1) / $1 * 100}' > target/criterion_u32_1024.txt
    (cat target/criterion_u32_1024.txt; cat target/tango.txt) > target/results.txt

(cat target/criterion.txt | grep 'change:' | awk 'OFS=";" {print "criterion/u32/1024", $3}'; cat target/tango.txt ) | tr -d '%' > target/results.txt
