#!/usr/bin/env bash

# This scripts is automating experiment on an AWS virtual machine
#
# The goal of an experiment is to measure performance variance reported by both harnesses (tango/criterion).
# UTF8 counting routine is used as a test function. The first one is counting up to 5000 characters in a string
# the second is up to 4950. We are expecting to see 1% difference in performance of those two functions

CRITERION=./target/criterion.txt
TANGO=./target/tango.txt
TANGO_FILTERED=./target/tango-filtered.txt

# Building and exporting all benchmarks. Align feature is used to disable inlining and to force 32-byte aligning
# of a tested functions. Without this trick the performance of the functions on Intel platform is heavily influenced
# by code aligning.
cargo +nightly export ./target/benchmarks -- bench --features=align --bench=criterion
cargo +nightly export target/benchmarks -- bench --features=align --bench='tango-*'

while :
do
    date | tee -a "${CRITERION}" | tee -a "${TANGO}" | tee -a "${TANGO_FILTERED}"

    # Running criterion benchmarks
    ./target/benchmarks/criterion --bench str_length_495 \
        --warm-up-time 1 --measurement-time 1 | tee -a "${CRITERION}"
    ./target/benchmarks/criterion --bench str_length_500 \
        --warm-up-time 1 --measurement-time 1 | tee -a "${CRITERION}"

    # Running tango benchmarks
    ./target/benchmarks/tango_faster compare ./target/benchmarks/tango_slower \
        -t 2000 -o -f 'str_length_limit' | tee -a "${TANGO}"
    ./target/benchmarks/tango_faster compare ./target/benchmarks/tango_slower \
        -t 2000 -f 'str_length_limit' | tee -a "${TANGO_FILTERED}"
done
