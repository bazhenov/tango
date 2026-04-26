#!/bin/bash
set -euo pipefail

date > result.txt
aws s3 cp result.txt s3://tango-exp-data/result.txt

shutdown -h now
