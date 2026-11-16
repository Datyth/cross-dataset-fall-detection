#!/usr/bin/env bash
set -euo pipefail
if [[ $(id -u) != 0 ]]; then
  echo 'Run this one-time Ubuntu build-tool setup as root.' >&2
  exit 1
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential python3-venv ca-certificates curl \
  libglib2.0-0 libgl1 libgomp1 pkg-config
