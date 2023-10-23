#!/usr/bin/env bash

FILE=./target/tango.txt

if [ -f "${FILE}" ]; then
    rm -f "${FILE}"
fi

for i in {1..30}; do
    cargo bench --bench=tango -- pair factorial_495 -t 1000 -v >> "${FILE}"
done

cat "${FILE}" | grep 'mean' | awk '{print $8}'
