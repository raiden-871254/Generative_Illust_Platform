#!/usr/bin/env bash
set -euo pipefail

run_container_mode() {
  local type="$1"
  local config_path="$2"
  local train_data_dir="$3"
  local set_name_arg="${4:-}"
  local prefix=""

  case "$type" in
    char) prefix="char_" ;;
    style) prefix="style_" ;;
    *)
      echo "Unknown type: $type (expected: char|style)" >&2
      exit 2
      ;;
  esac

  if [[ ! -d "$train_data_dir" ]]; then
    echo "train_data_dir not found: $train_data_dir" >&2
    exit 1
  fi

  local set_dir=""
  if [[ -n "$set_name_arg" ]]; then
    set_dir="${train_data_dir%/}/$set_name_arg"
    if [[ ! -d "$set_dir" ]]; then
      echo "Specified set not found: $set_dir" >&2
      exit 1
    fi
  else
    mapfile -t sets < <(find "$train_data_dir" -mindepth 1 -maxdepth 1 -type d | sort)

    if [[ "${#sets[@]}" -eq 0 ]]; then
      echo "No dataset folders found under: $train_data_dir" >&2
      exit 1
    fi

    if [[ "${#sets[@]}" -gt 1 ]]; then
      echo "Multiple dataset folders found under: $train_data_dir" >&2
      printf ' - %s\n' "${sets[@]}" >&2
      echo "Please keep only one set for this run, or pass set_name." >&2
      exit 1
    fi

    set_dir="${sets[0]}"
  fi

  local set_name
  local base_name
  set_name="$(basename "$set_dir")"
  base_name="$(echo "$set_name" | sed -E 's/^([0-9]+_+)//')"

  if [[ -z "$base_name" ]]; then
    base_name="$set_name"
  fi

  local output_name="${prefix}${base_name}"

  cd /opt/sd-scripts
  accelerate launch train_network.py \
    --config_file "$config_path" \
    --output_name "$output_name"
}

usage() {
  cat <<'USAGE'
Usage: run_lora.bash --profile <char|style> [--input <set_name>] [-- <docker compose args>]

Examples:
  ./scripts/run_lora.bash --profile char --input 1_cyrene
  ./scripts/run_lora.bash --profile style --input 1_ponnyu
  ./scripts/run_lora.bash --profile char -- --build
USAGE
}

if [[ $# -ge 1 && ( "$1" == "char" || "$1" == "style" ) ]]; then
  if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: run_lora <char|style> <config_path> <train_data_dir> [set_name]" >&2
    exit 2
  fi
  run_container_mode "$@"
  exit 0
fi

profile=""
input_set=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      profile="${2:-}"
      shift 2
      ;;
    --input)
      input_set="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$profile" ]]; then
  echo "--profile is required (char or style)" >&2
  usage
  exit 2
fi

if [[ "$profile" != "char" && "$profile" != "style" ]]; then
  echo "Invalid --profile: $profile" >&2
  usage
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
compose_dir="${script_dir}/../kohya-docker"
compose_file="${compose_dir}/docker-compose.yml"

if [[ ! -d "$compose_dir" ]]; then
  echo "kohya-docker not found: $compose_dir" >&2
  exit 1
fi
if [[ ! -f "$compose_file" ]]; then
  echo "docker-compose.yml not found: $compose_file" >&2
  exit 1
fi

if [[ -n "$input_set" ]]; then
  DATASET_SET="$input_set" docker compose -f "$compose_file" --project-directory "$compose_dir" --profile "$profile" up "${extra_args[@]}"
else
  docker compose -f "$compose_file" --project-directory "$compose_dir" --profile "$profile" up "${extra_args[@]}"
fi
