#!/bin/bash
set -euo pipefail

# Update packages
apt-get update -y
apt-get install -y curl git gcc screen
snap install aws-cli --classic

curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
echo "source ~/.cargo/env" >> ~/.bashrc
echo "source ~/.cargo/env" >> ~/.profile
source "$HOME/.cargo/env"

cargo install cargo-export
