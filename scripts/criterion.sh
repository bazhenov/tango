#!/usr/bin/env bash
set -eo pipefail

FILE=./target/criterion.txt

if [ -f "${FILE}" ]; then
    rm -f "${FILE}"
fi

for i in {1..30}; do
    cargo bench --bench=criterion str_length -- --warm-up-time 1 --measurement-time 1 >> "${FILE}"
done

for NAME in "str_length_500" "str_length_495"; do
    echo "${NAME}"
    cat "${FILE}" | grep -A1 "${NAME}" | grep 'time:' | awk '{print $5}'
done
