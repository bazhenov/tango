#!/usr/bin/env bash
set -eo pipefail

CRITERION=./target/criterion.txt
TANGO=./target/tango.txt
TANGO_FILTERED=./target/tango-filtered.txt

if [ "$1" == "tango" ]; then
    cat "${TANGO}" | awk '{print $(NF)}' | egrep -o '(-|\+)[0-9]+\.[0-9]+'
fi

if [ "$1" == "tango-filtered" ]; then
    cat "${TANGO_FILTERED}" | awk '{print $(NF)}' | egrep -o '(-|\+)[0-9]+\.[0-9]+'
fi

if [ "$1" == "criterion" ]; then
    paste \
        <(cat "${CRITERION}" | grep -A1 "str_length_5000" | grep 'time:' | awk '{print $5}') \
        <(cat "${CRITERION}" | grep -A1 "str_length_4950" | grep 'time:' | awk '{print $5}') | \
        awk '{print ($2 - $1) / $1 * 100}'
fi
