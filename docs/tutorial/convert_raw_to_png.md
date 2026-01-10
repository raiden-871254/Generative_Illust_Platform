# 画像フォーマット変換（convert_raw_to_png.py）

datasets/raw 配下の画像を PNG に変換し、ディレクトリ構成を維持したまま
datasets/raw_format に出力します。進捗はプログレスバーで表示されます。

## 使い方

```bash
python scripts/convert_raw_to_png.py --yes
```

## 例

```bash
python scripts/convert_raw_to_png.py --yes --jobs 24 --verify
```

- `--jobs` は並列変換数（デフォルト 24）。
- `--verify` で入力と出力の解像度一致を確認します。

## 仕様

- 入力の探索
  - 入力ディレクトリ配下を再帰的に探索します。
  - 対象拡張子: `.webp` `.jpg` `.jpeg` `.png`
- 出力
  - 入力のディレクトリ構成を維持して保存します。
  - 出力はすべて PNG です。
- 上書き
  - 既存ファイルがある場合、確認を求めます。
  - `--yes` を指定すると確認なしで上書きします。
  - `--jobs > 1` の場合は `--yes` が必須です。
- 解像度チェック
  - `--verify` 指定時に入力と出力のサイズを比較します。
  - 不一致数を集計して表示します。

## オプション

- `--input`: 入力ディレクトリ（デフォルト: `<project_root>/datasets/raw`）
- `--output`: 出力ディレクトリ（デフォルト: `<project_root>/datasets/raw_format`）
- `--jobs`: 並列プロセス数（デフォルト 24）
- `--verify`: 解像度一致を検証
- `-y`, `--yes`: 既存ファイルを確認なしで上書き
