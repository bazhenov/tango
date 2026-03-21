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

WORKDIR=./target/benches

mkdir -p "$WORKDIR"

# Exporting benchmarks
cargo export "$WORKDIR" -- bench -p tango-compare --bench=criterion --bench=tango

cd "$WORKDIR"

# Copying the executable to be more fair. On macOS trying to load executable as .dylib will
# lead to the same physical page mapped twice in VM of a process that will favor tango, because now
# we're testing not the 2 copies of the same code, but actually the same code, which is not fair
# to Criterion.
cp ./tango ./tango-1
cp ./tango ./tango-2

# Saving baseline
./criterion --save-baseline=main --bench -n > /dev/null

while true; do
    ./criterion --bench --baseline=main -n -v >> ./criterion.txt

    ./criterion --save-baseline=in-place --bench -n > /dev/null
    ./criterion --bench --baseline=in-place -n -v >> ./criterion-in-place.txt

    ./tango-1 compare ./tango-2 -t 1 -p >> ./tango.txt
    sleep 10
done
