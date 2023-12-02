#!/usr/bin/env bash

# This script runs tango-faster/tango-slower pair of benchmarks several times
# and reports how many times the results were statistically significant. Because
# those benchmarks was intentianlly constructed with performance difference,
# the bigger results the better (like [10/10]).

set -eo pipefail

cargo export target/benchmarks -- bench --bench='tango-*' --features=align

CMD="target/benchmarks/tango_faster compare target/benchmarks/tango_slower $@"
OUTPUT=""
ITERATIONS=10

for (( i=1; i<=ITERATIONS; i++ ))
do
    echo -n "."
    OUTPUT=$(paste <(echo "$OUTPUT") <($CMD))
done
echo

echo "Results:"
echo "$OUTPUT" | awk -v iter="$ITERATIONS" -F ' {2,}' '{printf(" [%3d/%3d] %s\n", gsub(/\*/,"", $0), iter, $1)}'
