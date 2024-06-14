#!/usr/bin/env bash

set -eo pipefail

cargo install cargo-export

cd hw_perf; make
