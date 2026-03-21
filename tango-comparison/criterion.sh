#!/usr/bin/env bash

set -eo pipefail

if [[ -f tango.txt ]]; then
    rm tango.txt
fi
if [[ -f criterion.txt ]]; then
    rm criterion.txt
fi
if [[ -f criterion-in-place.txt ]]; then
    rm criterion.txt
fi

# Exporting benchmarks
cargo export target/benches -- bench --bench=cr
cargo export target/benches -- bench --bench=tango-slower

cp ./target/benches/tango_slower ./target/benches/tango-1
cp ./target/benches/tango_slower ./target/benches/tango-2

# Saving baseline
./target/benches/cr --save-baseline=main --bench -n > /dev/null

while true; do
    ./target/benches/cr --bench --baseline=main -n -v --noise-threshold 0.001 >> criterion.txt

    ./target/benches/cr --save-baseline=in-place --bench -n > /dev/null
    ./target/benches/cr --bench --baseline=in-place -n -v --noise-threshold 0.001 >> criterion-in-place.txt

    ./target/benches/tango-1 compare ./target/benches/tango-2 -f 'str_length/random_limited' -t 1 -p >> tango.txt
    sleep 10
done


# cat criterion.txt | grep 'change:' | awk '{print $3}' | tr -d '%' > criterion.csv
# cat tango.txt | awk '{print $9}' | tr -d '%*' > tango.csv
#
# paste -d, \
#    <(cat criterion.txt | grep 'change:' | tail -n +2 | awk '{print $3}' | tr -d '%' | tr '−' '-') \
#    <(cat criterion-in-place.txt | grep 'change:' | tail -n +2 | awk '{print $3}' | tr -d '%' | tr '−' '-') \
#    <(cat tango.txt | awk '{print $9}' | tr -d '%*') > data.csv
