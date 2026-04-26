#!/bin/bash
set -euo pipefail

# Update packages
apt-get update -y
apt-get install -y curl git gcc screen
snap install aws-cli --classic
