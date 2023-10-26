#!/usr/bin/env bash

FILE=./target/tango.txt

if [ -f "${FILE}" ]; then
    rm -f "${FILE}"
fi

for i in {1..30}; do
    cargo bench --bench=tango -- pair std_495 -t 1000 -v -o >> "${FILE}"
done

cat "${FILE}" | grep 'mean' | awk '{print $10}' | sed 's/%//'
