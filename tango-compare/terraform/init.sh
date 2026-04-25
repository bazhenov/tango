#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/user-data.log) 2>&1

# Update packages
apt-get update -y
apt-get install -y curl git gcc screen
snap install aws-cli --classic

# Configure AWS CLI credentials for the default non-root user (ubuntu)
mkdir -p /home/ubuntu/.aws

cat > /home/ubuntu/.aws/credentials <<CREDS
[default]
aws_access_key_id = ${aws_access_key_id}
aws_secret_access_key = ${aws_secret_access_key}
CREDS

cat > /home/ubuntu/.aws/config <<CONF
[default]
region = ${aws_region}
output = json
CONF

chmod 600 /home/ubuntu/.aws/credentials /home/ubuntu/.aws/config
chown -R ubuntu:ubuntu /home/ubuntu/.aws

sudo -u ubuntu bash -ec '
    cd
    date > result.txt
    aws s3 cp result.txt s3://tango-exp-data/result.txt
'

shutdown -h now
