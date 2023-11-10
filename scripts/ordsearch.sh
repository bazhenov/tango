#!/usr/bin/env bash

set -eo pipefail

cargo +nightly export ./benches -- bench --bench=ordsearch

TARGET_FILE=target/ordsearch.txt

(for i in {1..30}; do
    ./benches/ordsearch pair "<u8, 8>" -o
done) | tee "$TARGET_FILE"

RESULT=$(paste \
    <(cat "$TARGET_FILE" | grep "1_" | awk '{print ($(NF)) + 0.0}') \
    <(cat "$TARGET_FILE" | grep "2_" | awk '{print ($(NF)) + 0.0}') \
    <(cat "$TARGET_FILE" | grep "3_" | awk '{print ($(NF)) + 0.0}'))

echo "$RESULT"

if [ -x "$(command -v pbcopy)" ]; then
    read -p "Copy to the clipboard [y/n]? " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "$RESULT" | pbcopy
    fi
fi
