#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
sudo chown -R raiden:raiden "${script_dir}/../datasets/"
sudo chown -R raiden:raiden "${script_dir}/../outputs/"
