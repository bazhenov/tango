#!/bin/bash
set -euo pipefail

git clone -b main --depth=1 https://github.com/bazhenov/tango.git tango-main
git clone -b shmem-test --depth=1 https://github.com/bazhenov/tango.git tango-commpage

cargo export . -t main -- bench --manifest-path tango-main/Cargo.toml --bench=tango-slower
cargo export . -t commpage -- bench --manifest-path tango-commpage/Cargo.toml --bench=tango-slower

UUID=$(uuidgen)
echo "Running experiment $UUID"

# Uploading archive with metadata only to indicate experiment is in progress
mkdir result
cat /proc/cpuinfo > result/cpuinfo
hostname > result/hostname
free > result/free
# reading DMI product name (for AWS this is instance type eg. t4g.small)
if [[ -f /sys/devices/virtual/dmi/id/product_name ]]; then
    cat /sys/devices/virtual/dmi/id/product_name > result/aws_instance_type
fi

tar czvf result.tar.gz result/
aws s3 cp result.tar.gz "s3://${s3_bucket_name}/$UUID.tar.gz"

for i in $(seq 1 1000);
do
    (./tango_slower-commpage compare -f factorial -s 100 || true) >> result/commpage.txt
    (./tango_slower-main compare -f factorial --sampler flat -s 100 -p || true) >> result/main.txt
done

tar czvf result.tar.gz result/

echo "Uploading test result $UUID"
aws s3 cp result.tar.gz "s3://${s3_bucket_name}/$UUID.tar.gz"
