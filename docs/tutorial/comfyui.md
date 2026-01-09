# ComfyUI（Docker）

Docker ComposeでComfyUIを起動し、設定ファイルや出力先を運用するためのメモです。

## 起動方法

```bash
docker compose up
```

停止する場合は以下を実行します。

```bash
docker compose down
```

## 設定ファイル

GUI用の設定（ワークフロー例）は `configs/gui` にあります。
必要に応じてJSONを読み込んで利用してください。

## ディレクトリ構成（要点）

- `output`
  - 生成結果の保存先
- `model`
  - モデルや関連ファイルの配置先

