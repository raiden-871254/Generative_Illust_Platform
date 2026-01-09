# 画像リサイズ（resize_dataset.py）

LoRA学習向けに画像を前処理するためのスクリプトです。
アスペクト比や最小サイズの条件を満たす画像のみを採用し、
短辺/長辺を保ちながらサイズを調整して出力します。

## 使い方

```bash
python scripts/resize_dataset.py \
  --input datasets/raw \
  --output normalized \
  --target-long 768 \
  --min-short 512 \
  --max-ar 2.2 \
  --convert-rgb
```

- `--input` を省略すると `datasets/raw` が使われます。
- `--output` を省略すると `normalized` が使われます。
- 出力先が存在しない場合は自動で作成されます。

## 仕様

- 画像の探索
  - 入力ディレクトリ配下を再帰的に探索します。
  - 対象拡張子: `.png` `.jpg` `.jpeg` `.webp` `.bmp` `.tif` `.tiff`
- 画像の検査
  - 最小辺が `min-short` 未満の画像は除外します。
  - アスペクト比が `1/max-ar`〜`max-ar` の範囲外の画像は除外します。
- リサイズ
  - 長辺が `target-long` に近づくようにアスペクト比を維持して拡大縮小します。
  - 幅・高さを `multiple` の倍数（デフォルト 64）に丸めます。
  - リサンプルは `LANCZOS` を使用します。
- 出力
  - 入力のディレクトリ構成を維持して `output` に保存します。
  - 入力が PNG の場合は PNG のまま保存します。
  - それ以外は JPG で保存します（品質 95、progressive）。
- 例外・拒否ログ
  - `preprocess_log.csv` を出力先直下に作成します。
  - `--rejected` を指定すると、除外された画像を指定先にコピーします。

## オプション

- `--input`: 入力ディレクトリ（再帰的に探索）
- `--output`: 出力ディレクトリ
- `--target-long`: 長辺の目標サイズ（デフォルト 768）
- `--multiple`: 出力サイズの丸め単位（デフォルト 64）
- `--min-short`: 最小辺の下限（デフォルト 512）
- `--max-ar`: 許容アスペクト比の上限（デフォルト 2.2）
- `--convert-rgb`: RGB/RGBAに変換して保存（モード問題の回避用）
- `--rejected`: 除外画像のコピー先

