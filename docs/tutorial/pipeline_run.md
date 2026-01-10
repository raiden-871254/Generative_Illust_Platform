# pipeline_run.py チュートリアル

## 目的

`scripts/pipeline_run.py` で以下を一括自動化します。

1) 初回データ投入時のリサイズ（`resize_dataset.py`）
2) LoRA学習（`run_lora.bash` でkohya起動）
3) 学習LoRAを使った固定条件ベンチ生成（ComfyUI）
4) LoRA weight の自動スイープ
5) 生成結果の定量指標と比較用グリッドの出力

## 前提条件

### uv / Python

- Python 3.10+ を使用
- 実行は `uv run python ...` 推奨（`python3` 直叩きは環境ズレの可能性あり）

### 必要ライブラリ

```
uv add pillow
```

### Docker

- ComfyUI は事前起動が必要
- kohya は `scripts/run_lora.bash` が起動するため事前 `up` 不要

ComfyUI 起動例:

```
docker compose -f comfy-docker/docker-compose.yml up -d
```

## ディレクトリ構造（入力・出力）

- 入力
  - `datasets/raw/style/<set>`（style LoRA 入力）
- 中間
  - `datasets/normalized/...`（リサイズ済み）
- 学習済みLoRAの正本
  - `output/kohya/<style|char>_<base>__<run_id>.safetensors`
- ログ・作業ファイル
  - `output/logs/<run_id>/...`（rejected / preprocess_log など）
- ComfyUI 出力
  - `output/bench/<run_id>/...`（ベンチ生成の元画像）
- ベンチ評価
  - `output/bench/<run_id>/grid/*.png`（最優先で見るグリッド）
  - `output/bench/<run_id>/metrics.csv`（補助の定量指標）

## クイックスタート（最短）

```
uv run python scripts/pipeline_run.py \
  --profile style \
  --raw-set ponnyu_v1 \
  --resize --yes \
  --bench-workflow configs/gui/MeinaMix_v12_lora_bench.json \
  --weights 0.4,0.6,0.8 \
  --seeds 123456789,987654321 \
  --comfy-url http://127.0.0.1:8188
```

## 引数リファレンス（pipeline_run.py）

必須:

- `--raw-set`（必須）: `datasets/raw/{style|characters}/<set>` のフォルダ名  
  `--skip-resize` を使う場合は `datasets/normalized/{style|characters}/<set>` を指定

任意（デフォルトあり）:

- `--profile`（default: `style`）: `style` or `char`
- `--run-id`（default: 現在時刻 `YYYYMMDD_HHMMSS`）
- `--weights`（default: `0.4,0.6,0.8`）
- `--seeds`（default: `123456789`）
- `--bench-workflow`（default: `configs/gui/MeinaMix_v12_lora_bench.json`）
- `--comfy-url`（default: `http://127.0.0.1:8188`）
- `--positive`（default: `None`）: 既存workflowの正/負プロンプトを上書き
- `--negative`（default: `None`）
- `--batch-size`（default: `1`）: ベンチ用のバッチサイズ
- `--lora-install-dir`（default: `models/loras`）
- `--run-lora-bash`（default: `scripts/run_lora.bash`）
- `--resize-script`（default: `scripts/resize_dataset.py`）
- `--fix-owner-bash`（default: `scripts/fix_datasets_owner.bash`）

フラグ:

- `--resize`: 事前リサイズを実行
- `--yes`: リサイズの上書き確認をスキップ
- `--skip-train`: kohya学習をスキップ（ベンチのみ再実行）
- `--skip-bench`: ベンチ生成をスキップ（前処理/学習のみ）
- `--skip-resize`: `datasets/normalized` を直接利用（raw→normalized推定を省略）
- `--bench-only`: デバッグ用。学習とリサイズをスキップしてベンチのみ実行

## 実行フロー

1) `scripts/fix_datasets_owner.bash` を実行  
2) `--resize` 指定時に `scripts/resize_dataset.py` を実行  
   - `--convert-rgb` を付与  
   - `datasets/raw` → `datasets/normalized`  
   - 除外画像は `output/logs/<run_id>/rejected/` に保存  
   - `datasets/normalized/preprocess_log.csv` を `output/logs/<run_id>/preprocess_log.csv` にスナップショット  
3) `preprocess_log.csv` から正規化済みセット名を推定  
4) `scripts/run_lora.bash` でkohya学習  
   - 学習結果は `output/kohya/<style|char>_<base>.safetensors`  
   - 正本は `output/kohya/<style|char>_<base>__<run_id>.safetensors` に保存  
   - ComfyUI用に `models/loras/` へコピー  
5) ComfyUIでベンチ生成（weight/seedスイープ）  
6) `output/bench/<run_id>/metrics.csv` と `output/bench/<run_id>/grid/*.png` を出力

## 生成結果の見方

- 最優先: `output/bench/<run_id>/grid/*.png`  
  例: `output/bench/20240101_123456/grid/seed123456789_weights.png`
- 補助: `output/bench/<run_id>/metrics.csv`
- 生成画像: `output/bench/<run_id>/w0.40/seed123456789_*.png`

## よくあるトラブルと対処

- ComfyUIに繋がらない  
  - `--comfy-url` のURL/ポートを確認  
  - `docker compose -f comfy-docker/docker-compose.yml up -d` で起動
- LoRAが見つからない  
  - `models/loras` にLoRAが配置されているか確認  
  - ComfyUIの探索パスが `models/loras` を含むか確認
- outputに画像が無い  
  - ComfyUIの `output` マウントが `./output` に向いているか確認  
  - `comfy-docker/docker-compose.yml` のvolume設定を確認
- preprocess_log.csvが見つからない  
  - `--resize` を実行したか確認  
  - `datasets/normalized/preprocess_log.csv` の有無を確認  
  - もしくは `--skip-resize` で normalized 名を直接指定

## 運用Tips

- style v1の目安: 35枚程度  
- weights: `0.4,0.6,0.8`  
- seeds: 固定（例: `123456789,987654321`）  
- raw-set の命名例: `datasets/raw/style/ponnyu_v1/`

## 注意点（重要）

- bench workflow JSON は原則編集不要  
  - weight/seed/SaveImage prefixは実行時に自動パッチされます
- 実行は `uv run python ...` 推奨
- ComfyUIは事前起動が必要、kohyaは不要（`run_lora.bash` が起動）
- 出力の正本は `output/kohya/`（上書き防止）
