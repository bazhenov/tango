#!/bin/bash
set -euo pipefail

git clone -b main --depth=1 https://github.com/bazhenov/tango.git tango

cargo export bin -- bench -p tango-compare --manifest-path tango/Cargo.toml --bench=criterion
cargo export bin -- bench -p tango-compare --manifest-path tango/Cargo.toml --bench=tango

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

# Saving criterion baseline
./bin/criterion --save-baseline master --bench binary_search || true

for i in $(seq 1 1000);
do
    (./bin/criterion --baseline master --bench binary_search --measurement-time 1 --warm-up-time 0.1 || true) >> result/criterion.txt
    (./bin/tango compare -t 1 -f binary_search || true) >> result/tango.txt
done

tar czvf result.tar.gz result/

echo "Uploading test result $UUID"
aws s3 cp result.tar.gz "s3://${s3_bucket_name}/$UUID.tar.gz"
