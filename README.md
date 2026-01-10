# Generative_Illust_Platform

Stable Diffusionでイラストを生成するための汎用プラットフォーム

# Style LoRAベンチ自動パイプライン（最小構成）

## ファイル
- `pipeline_run.py` : 前処理 -> 学習 -> ベンチ -> 指標/グリッドを統合実行
- `comfy_client.py` : ComfyUI HTTP APIの送信とポーリング
- `bench_utils.py` : 簡易指標とコンタクトシート生成

## インストール
以下のようにリポジトリ配下へ配置してください:

```
scripts/pipeline_run.py
scripts/comfy_client.py
scripts/bench_utils.py
```

必要要件:
- Python 3.10+
- Pillow (`uv add pillow`)

## 実行例

### フル実行（推奨）
```
python scripts/pipeline_run.py \
  --profile style \
  --raw-set my_style_set \
  --resize --yes \
  --fix-owner \
  --bench-workflow configs/gui/MeinaMix_v12_lora_bench.json \
  --weights 0.4,0.6,0.8 \
  --seeds 123456789,987654321 \
  --comfy-url http://127.0.0.1:8188
```

### ベンチのみ再実行（weight/プロンプト調整後）
```
python scripts/pipeline_run.py \
  --profile style \
  --raw-set my_style_set \
  --skip-train \
  --bench-workflow configs/gui/MeinaMix_v12_lora_bench.json \
  --weights 0.2,0.4,0.6,0.8,1.0 \
  --seeds 123456789
```

## 事前に確認すべきメモ / 前提
1) ComfyUIの出力がリポジトリの`./output/`にマッピングされていること（ホスト側で生成画像を参照するため）。
2) ComfyUIが`models/loras`からLoRAを読み込めること（このスクリプトの既定インストール先）。
3) 現在のcompose内の`run_lora.bash`はデータセット名から`output_name`を決定し、次回実行で同名を上書きします。
   このパイプラインは学習済みLoRAを`runs/<run_id>/train/`にrun_id付きでアーカイブし、そのファイル名をComfyUI用に`models/loras`へ配置します。
