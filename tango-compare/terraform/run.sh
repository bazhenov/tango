#!/bin/bash
set -euo pipefail

git clone -b main --depth=1 https://github.com/bazhenov/tango.git tango-main
git clone -b shmem-test --depth=1 https://github.com/bazhenov/tango.git tango-commpage

mkdir result

cargo export . -t main -- bench --manifest-path tango-main/Cargo.toml --bench=tango-slower
cargo export . -t commpage -- bench --manifest-path tango-commpage/Cargo.toml --bench=tango-slower

UUID=$(uuidgen)
echo "Running experiment $UUID"

# Uploading empty archive to indicate experiment is in progress
touch result.tar.gz
aws s3 cp result.tar.gz "s3://${s3_bucket_name}/$UUID.tar.gz"

for i in $(seq 1 1000);
do
    (./tango_slower-commpage compare -f factorial -t 1 || true) >> result/commpage.txt
    (./tango_slower-main compare -f factorial --sampler flat -t 1 -p || true) >> result/main.txt
done

tar czvf result.tar.gz result/

echo "Uploading test result $UUID"
aws s3 cp result.tar.gz "s3://${s3_bucket_name}/$UUID.tar.gz"
