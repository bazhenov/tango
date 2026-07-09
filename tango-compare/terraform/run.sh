#!/bin/bash
set -euo pipefail

git clone -b criterion-to-tango-comparison --depth=1 https://github.com/bazhenov/tango.git tango

cargo export bin -- bench -p tango-compare --manifest-path tango/Cargo.toml --bench=criterion
cargo export bin -- bench -p tango-compare --manifest-path tango/Cargo.toml --bench=tango

BENCH=hash
UUID=$(uuidgen)
echo "Running experiment $UUID"

# Uploading archive with metadata only to indicate experiment is in progress
mkdir result
cat /proc/cpuinfo > result/cpuinfo
hostname > result/hostname
free > result/free

# Copying run script itself to have a reference what exactly was run
cp $0 result
# reading DMI product name (for AWS this is instance type eg. t4g.small)
if [[ -f /sys/devices/virtual/dmi/id/product_name ]]; then
    cat /sys/devices/virtual/dmi/id/product_name > result/aws_instance_type
fi

tar czvf result.tar.gz result/
aws s3 cp result.tar.gz "s3://${s3_bucket_name}/$UUID.tar.gz"

# Saving criterion baseline
./bin/criterion --save-baseline master --bench "$BENCH" || true

for i in $(seq 1 1000);
do
    # Running criterion in classical mode
    (./bin/criterion --baseline master --bench "$BENCH" --measurement-time 1 --warm-up-time 0.1 || true) >> result/criterion.txt

    # Running criterion in psuedo pairwise mode
    (./bin/criterion --save-baseline pairwise --bench "$BENCH" --measurement-time 1 --warm-up-time 0.1 || true)
    (./bin/criterion --baseline pairwise --bench "$BENCH" --measurement-time 1 --warm-up-time 0.1 || true) >> result/criterion-pairwise.txt

    (./bin/tango compare -t 1 -f "$BENCH" || true) >> result/tango.txt
done

tar czvf result.tar.gz result/

echo "Uploading test result $UUID"
aws s3 cp result.tar.gz "s3://${s3_bucket_name}/$UUID.tar.gz"
