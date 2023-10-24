#!/usr/bin/env bash

FILE=./target/criterion.txt

if [ -f "${FILE}" ]; then
    rm -f "${FILE}"
fi

for i in {1..30}; do
    cargo bench --bench=criterion factorial -- --warm-up-time 1 --measurement-time 1 >> "${FILE}"
done

echo "factorial_500"
cat "${FILE}" | grep -A1 'factorial_500' | grep 'time:' | awk '{print $4}'

echo "factorial_495"
cat "${FILE}" | grep -A1 'factorial_495' | grep 'time:' | awk '{print $4}'
