#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: run_lora <char|style> <config_path> <train_data_dir> [set_name]" >&2
  exit 2
fi

type="$1"
config_path="$2"
train_data_dir="$3"
set_name_arg="${4:-}"

case "$type" in
  char) prefix="char_" ;;
  style) prefix="style_" ;;
  *)
    echo "Unknown type: $type (expected: char|style)" >&2
    exit 2
    ;;
esac

if [ ! -d "$train_data_dir" ]; then
  echo "train_data_dir not found: $train_data_dir" >&2
  exit 1
fi

if [ -n "$set_name_arg" ]; then
  set_dir="${train_data_dir%/}/$set_name_arg"
  if [ ! -d "$set_dir" ]; then
    echo "Specified set not found: $set_dir" >&2
    exit 1
  fi
else
  mapfile -t sets < <(find "$train_data_dir" -mindepth 1 -maxdepth 1 -type d | sort)

  if [ "${#sets[@]}" -eq 0 ]; then
    echo "No dataset folders found under: $train_data_dir" >&2
    exit 1
  fi

  if [ "${#sets[@]}" -gt 1 ]; then
    echo "Multiple dataset folders found under: $train_data_dir" >&2
    printf ' - %s\n' "${sets[@]}" >&2
    echo "Please keep only one set for this run, or pass set_name." >&2
    exit 1
  fi

  set_dir="${sets[0]}"
fi
set_name="$(basename "$set_dir")"
base_name="$(echo "$set_name" | sed -E 's/^([0-9]+_+)//')"

if [ -z "$base_name" ]; then
  base_name="$set_name"
fi

output_name="${prefix}${base_name}"

cd /opt/sd-scripts
accelerate launch train_network.py \
  --config_file "$config_path" \
  --output_name "$output_name"
