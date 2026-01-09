#!/usr/bin/env bash
set -euo pipefail

# scripts/kohya/train_lora.bash
# 使い方:
#   docker compose -f kohya-docker/docker-compose.yml run --rm kohya \
#     bash -lc "./scripts/kohya/train_lora.bash"

SD_SCRIPTS_DIR="/opt/sd-scripts"

# 例: SD1.5 / SDXL どちらでも「ベースモデルのパス」は差し替えて使う
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/workspace/models/checkpoints/meinamix_v12Final.safetensors}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/workspace/datasets/normalized}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/kohya}"
OUTPUT_NAME="${OUTPUT_NAME:-my_lora}"

mkdir -p "${OUTPUT_DIR}"

cd "${SD_SCRIPTS_DIR}"

# accelerate config を対話無しで済ませたい場合:
# 1) まず一度だけコンテナ内で `accelerate config` を実行して設定ファイルを作る
# 2) それを /workspace/.cache などに置いて使い回す
#
# ここでは簡易にデフォルトで走らせる（必要なら --config_file を追加）
accelerate launch train_network.py \
  --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
  --train_data_dir="${TRAIN_DATA_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --output_name="${OUTPUT_NAME}" \
  --resolution=512,512 \
  --network_module=networks.lora \
  --network_dim=16 \
  --train_batch_size=1 \
  --max_train_epochs=10 \
  --learning_rate=1e-4 \
  --optimizer_type=AdamW8bit \
  --mixed_precision=fp16 \
  --save_model_as=safetensors \
  --sdpa
