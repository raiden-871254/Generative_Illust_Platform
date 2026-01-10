# bench_sweep.py チュートリアル

ComfyUI を使って LoRA のベンチ生成を行うスクリプトです。  
`stability`（固定 seed で比較）と `harvest`（ランダム seed で大量生成）の2モードを持ちます。

## 前提

- Python 3.10+
- ComfyUI が利用可能（自動起動/停止に対応）
- 対象 LoRA を `models/loras/` に配置済み
- workflow は ComfyUI API 形式（`configs/gui/MeinaMix_v12_lora_bench.json`）

## クイックスタート

### stability（固定 seed 比較）

```
uv run python scripts/bench_sweep.py \
  --mode stability \
  --lora-file style_cocomiya_bench.safetensors \
  --weights 0.4,0.6,0.8 \
  --seeds 111,222,333,444,555,666,777,888,999,123456789,987654321,314159265 \
  --batch-size 1
```

### harvest（大量生成）

```
uv run python scripts/bench_sweep.py \
  --mode harvest \
  --lora-file style_cocomiya_bench.safetensors \
  --weights 0.6 \
  --seeds 123456789,987654321 \
  --batch-size 1
  --harvest-count 30
```

## 出力先

- stability:
  - 画像: `output/bench/stability/<run_id>/w*/seed*/...`
  - 指標: `output/bench/stability/<run_id>/metrics.csv`
  - グリッド: `output/bench/stability/<run_id>/grid/*.png`
- harvest:
  - 画像: `output/bench_runs/harvest/<run_id>/w*/seed*/...`
  - 指標: `output/bench/stability/<run_id>/metrics.csv`（※現状、metricsは stability 配下に出力）

## 引数リファレンス

必須:

- `--mode`: `stability` or `harvest`
- `--lora-file`: `models/loras/` 配下のファイル名

任意（デフォルトあり）:

- `--run-id`（default: 現在時刻 `YYYYMMDD_HHMMSS`）
- `--weights`（default: `0.6`）: LoRAのweight一覧（カンマ区切り）
- `--seeds`（default: `123456789,987654321`）: stability用のseed一覧（harvestでは基本無視）
- `--batch-size`（default: `1`）: 1回の生成で出力する枚数
- `--harvest-count`（default: `50`）: harvestで何回投げるか（回数×batch-sizeが総枚数）
- `--harvest-seed-min`（default: `1`）: harvestのseed下限
- `--harvest-seed-max`（default: `2000000000`）: harvestのseed上限
- `--workflow`（default: `configs/gui/MeinaMix_v12_lora_bench.json`）: ComfyUI API形式のworkflow
- `--positive` / `--negative`（default: `None`）: workflowのプロンプトを上書き
- `--out-root`（default: `output/bench_runs`）: harvest出力のルート（prefixに使われる）

ComfyUI関連:

- `--comfy-url`（default: `http://127.0.0.1:8188`）: ComfyUIの接続先
- `--comfy-compose-file`（default: `comfy-docker/docker-compose.yml`）: docker composeのパス
- `--comfy-service`（default: `comfyui`）: 起動/停止対象のサービス名
- `--comfy-wait-timeout`（default: `180`）: ComfyUI待機時間（秒）
- `--no-start-comfy`: 自動起動を無効化（既に起動済みの場合向け）
- `--no-stop-comfy-after`: 自動停止を無効化（継続利用する場合向け）

## 動作の流れ

1) ComfyUI を自動起動（`--no-start-comfy` で無効化）
2) workflow を読み込み、LoRA/seed/weight を差し替え
3) 生成結果を `output/` 配下に保存
4) metrics.csv とグリッド画像を出力
5) ComfyUI を自動停止（`--no-stop-comfy-after` で無効化）

## 生成結果の見方

- 最優先: `output/bench/stability/<run_id>/grid/*.png`
- 補助: `output/bench/stability/<run_id>/metrics.csv`
- 画像: `output/bench/stability/<run_id>/w0.60/seed123456789_*.png`

## よくあるトラブル

- LoRA が見つからない  
  - `models/loras/<lora-file>` の存在を確認
- ComfyUI に繋がらない  
  - `--comfy-url` を確認  
  - `--no-start-comfy` 使用時は手動で `docker compose up -d`
*** End Patch
