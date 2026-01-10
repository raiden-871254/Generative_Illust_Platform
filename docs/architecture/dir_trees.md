# ディレクトリ構成

以下はリポジトリ直下の構成です（`.git` `.venv` `.mypy_cache` は省略、生成物は一部のみ記載）。

```text
.
├── .github
│   ├── ISSUE_TEMPLATE
│   │   ├── bug.md
│   │   ├── chore.md
│   │   ├── config.yml
│   │   ├── docs.md
│   │   ├── feature.md
│   │   ├── hotfix.md
│   │   └── release.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── extensions
│       └── extensions.txt
├── .cache
│   └── huggingface
│       ├── .locks
│       ├── hub
│       ├── models--openai--clip-vit-large-patch14
│       └── version.txt
├── .gitignore
├── .python-version
├── README.md
├── comfy-docker
│   ├── custom_nodes
│   │   ├── example_node.py.example
│   │   └── websocket_image_save.py
│   └── docker-compose.yml
├── configs
│   ├── gui
│   │   ├── MeinaMix_v12_lora_0_4_bench.json
│   │   ├── MeinaMix_v12_lora_0_8_bench.json
│   │   ├── MeinaMix_v12_lora_bench.json
│   │   ├── MeinaMix_v12_lora_sample.json
│   │   ├── MeinaMix_v12_sample.json
│   └── lora
│       ├── char_lora.toml
│       └── style_lora.toml
├── datasets
│   ├── normalized
│   │   ├── characters
│   │   ├── style
│   │   └── preprocess_log.csv
│   ├── pickied
│   ├── raw
│   │   ├── characters
│   │   └── style
│   └── staging
│       └── _STAGING_INFO.json
├── docs
│   ├── architecture
│   │   └── dir_trees.md
│   ├── design
│   │   └── SPEC.md
│   ├── rules
│   │   └── datasets.md
│   └── tutorial
│       ├── comfyui.md
│       ├── image_resize.md
│       └── kohya.md
├── kohya-docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── logs
│   └── kohya-docker.log
├── lora
│   ├── releases
│   └── working
├── main.py
├── models
│   ├── checkpoints
│   │   └── meinamix_v12Final.safetensors
│   ├── loras
│   └── vae
├── output
│   ├── bench
│   ├── kohya
│   │   └── char_cyrene.safetensors
│   └── logs
│       └── ...
├── pyproject.toml
├── scripts
│   ├── fix_datasets_owner.bash
│   ├── resize_dataset.py
│   └── run_lora.bash
└── uv.lock
```
