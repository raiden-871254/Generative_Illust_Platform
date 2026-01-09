# kohya（Docker）

kohya-ss/sd-scripts を Docker で動かす手順と、設定ファイルの要点をまとめます。

## Docker ビルド

```bash
cd kohya-docker

docker compose --profile build build
```

> `profile` を使っているため、通常の `docker compose build` では対象がなく警告になります。

## 起動方法

```bash
# Character LoRA
docker compose --profile char up

# Style LoRA
docker compose --profile style up
```

停止は `Ctrl+C` か `docker compose --profile char down` を使用します。

## 設定ファイル

- Character: `configs/lora/char_lora.toml`
- Style: `configs/lora/style_lora.toml`

Docker では以下のパスを前提にしています。

- データセット: `/workspace/datasets`
- モデル: `/workspace/models`
- 出力: `/workspace/output`
- 設定: `/workspace/configs`

## config パラメータ（主要項目）

- `pretrained_model_name_or_path`
  - 学習元モデルのパス（例: `/workspace/models/checkpoints/xxx.safetensors`）
- `train_data_dir`
  - 学習データの親ディレクトリ
  - 直下にクラス別フォルダがある構成が必要
- `output_dir`
  - 学習結果の出力先
- `logging_dir`
  - ログ出力先
- `output_name`
  - 出力ファイル名のベース
- `resolution`
  - 学習解像度（必須）
- `train_batch_size`
  - バッチサイズ
- `num_epochs`
  - エポック数
- `network_module`
  - LoRA 実装（通常は `networks.lora`）
- `network_dim` / `network_alpha`
  - LoRA のランク/alpha
- `learning_rate` / `unet_lr` / `text_encoder_lr`
  - 学習率設定
- `optimizer_type`
  - 例: `AdamW`
- `mixed_precision` / `save_precision`
  - 例: `fp16`
- `enable_bucket`
  - 画像サイズが混在する場合に有効
- `caption_extension`
  - キャプション拡張子（例: `.txt`）
- `console_log_interval`
  - 進捗ログの出力間隔（例: `10`）

## データセット構成例

```text
datasets/normalized/characters/
  ├── 1_character_name/
  │   ├── image_001.jpg
  │   └── image_001.txt
  └── 2_other_character/
```

`train_data_dir` には、画像フォルダの親ディレクトリを指定します。

