#!/usr/bin/env bash

set -eo pipefail

WORKDIR=.
cd "$WORKDIR"

paste -d, \
    <(cat "$WORKDIR/criterion.txt" | grep 'change:' | tail -n +2 | awk '{print $3}' | tr -d '%' | tr '−' '-') \
    <(cat "$WORKDIR/criterion-in-place.txt" | grep 'change:' | tail -n +2 | awk '{print $3}' | tr -d '%' | tr '−' '-') \
    <(cat "$WORKDIR/tango.txt" | awk '{print $9}' | tr -d '%*') > data.csv

gnuplot criterion-vs-tango.gnuplot
