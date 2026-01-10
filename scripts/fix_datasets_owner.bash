#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
sudo chown -R raiden:raiden "${script_dir}/../datasets/"
sudo chown -R raiden:raiden "${script_dir}/../output/"
sudo chown -R raiden:raiden "${script_dir}/../output/bench/"
sudo chown -R raiden:raiden "${script_dir}/../output/logs/"
