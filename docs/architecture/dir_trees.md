# ディレクトリ構成

以下はリポジトリ直下の構成です（`.git` `.venv` `.mypy_cache` は省略）。

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
├── .gitignore
├── .python-version
├── README.md
├── comfy-docker
│   ├── custom_nodes
│   │   └── ComfyUI-Manager
│   └── docker-compose.yml
├── configs
│   └── gui
│       └── MeinaMix_v12_sample.json
├── datasets
│   ├── normalized
│   ├── pickied
│   ├── raw
│   └── staging
├── docs
│   ├── architecture
│   │   └── dir_trees.md
│   ├── rules
│   │   └── datasets.md
│   └── tutorial
│       ├── comfyui.md
│       └── image_resize.md
├── main.py
├── models
│   ├── checkpoints
│   │   └── meinamix_v12Final.safetensors
│   ├── lora
│   └── vae
├── output
├── pyproject.toml
├── scripts
│   ├── fix_datasets_owner.bash
│   └── resize_dataset.py
└── uv.lock
```

