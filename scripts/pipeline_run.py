#!/usr/bin/env python3
"""
One-shot automation for:
1) resize_dataset.py (raw -> normalized)
2) kohya training via run_lora.bash (style profile)
3) ComfyUI benchmark generation with fixed seeds/prompts
4) LoRA weight sweep
5) Basic metrics + contact sheets for quick visual selection

Expected to be run from repo root.

Example:
  python scripts/pipeline_run.py --profile style --raw-set my_new_set \
    --weights 0.4,0.6,0.8 --seeds 123456789,987654321 \
    --bench-workflow configs/gui/MeinaMix_v12_lora_bench.json \
    --comfy-url http://127.0.0.1:8188

Notes:
- This script patches the bench workflow JSON at runtime:
  - LoraLoader: filename + strength_model/clip
  - KSampler: seed + control ('fixed')
  - EmptyLatentImage: batch_size=1 (recommended for bench)
  - SaveImage: filename_prefix with run_id + weight + seed
  - (Optional) CLIPTextEncode prompts
- run_lora.bash inside the container derives output_name from the dataset set:
  style: style_<base_set_name_without_numeric_prefix>
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from comfy_client import ComfyClient
from bench_utils import compute_metrics, write_metrics_csv, make_contact_sheet


REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def parse_normalized_set_name(preprocess_log: Path, raw_set: str, group: str = "style") -> str:
    """
    Determine normalized topdir name for a given raw set by reading preprocess_log.csv,
    finding first OK row under datasets/raw/{group}/{raw_set}/..., and extracting rel = "{group}/{normalized_topdir}/..."
    """
    if not preprocess_log.exists():
        raise FileNotFoundError(f"preprocess_log not found: {preprocess_log}")

    raw_pat = f"{os.sep}datasets{os.sep}raw{os.sep}{group}{os.sep}{raw_set}{os.sep}"
    with preprocess_log.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("src", "")
            ok = row.get("ok", "0") == "1"
            rel = row.get("rel", "")
            if ok and raw_pat in src and rel.startswith(f"{group}/"):
                parts = Path(rel).parts
                if len(parts) >= 2:
                    return parts[1]
    raise RuntimeError(
        f"Could not infer normalized set dir for raw_set='{raw_set}'. "
        f"Check that datasets/raw/{group}/{raw_set}/ contains images and preprocess succeeded."
    )


def base_name_from_set(set_name: str) -> str:
    # strip leading digits/underscores: "20_cyrene" -> "cyrene"
    return re.sub(r"^([0-9]+_+)", "", set_name) or set_name


def patch_workflow_for_bench(
    workflow: Dict[str, Any],
    *,
    lora_filename: str,
    lora_weight: float,
    seed: int,
    filename_prefix: str,
    positive: Optional[str] = None,
    negative: Optional[str] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Patch specific nodes in the provided workflow:
      - LoraLoader (id=11 in your bench file): widgets_values = [filename, strength_model, strength_clip]
      - KSampler (id=6): widgets_values[0]=seed, widgets_values[1]='fixed'
      - EmptyLatentImage (id=5): widgets_values[2]=batch_size
      - SaveImage (id=7): widgets_values[0]=filename_prefix
      - CLIPTextEncode positive/negative (ids=3 and 4): widgets_values[0]=text
    """
    wf = json.loads(json.dumps(workflow))  # deep copy
    nodes = {n["id"]: n for n in wf.get("nodes", [])}

    def must_node(nid: int) -> Dict[str, Any]:
        if nid not in nodes:
            raise KeyError(f"Node id {nid} not found in workflow")
        return nodes[nid]

    # LoraLoader
    lora = must_node(11)
    wv = lora.get("widgets_values", [])
    if len(wv) < 3:
        raise RuntimeError(f"Unexpected LoraLoader widgets_values: {wv}")
    wv[0] = lora_filename
    wv[1] = float(lora_weight)
    wv[2] = float(lora_weight)
    lora["widgets_values"] = wv

    # KSampler seed + fixed
    ks = must_node(6)
    ksw = ks.get("widgets_values", [])
    if len(ksw) < 2:
        raise RuntimeError(f"Unexpected KSampler widgets_values: {ksw}")
    ksw[0] = int(seed)
    ksw[1] = "fixed"
    ks["widgets_values"] = ksw

    # EmptyLatentImage batch size
    el = must_node(5)
    elw = el.get("widgets_values", [])
    if len(elw) >= 3:
        elw[2] = int(batch_size)
        el["widgets_values"] = elw

    # SaveImage prefix (supports subfolders)
    si = must_node(7)
    siw = si.get("widgets_values", [])
    if not siw:
        siw = [filename_prefix]
    else:
        siw[0] = filename_prefix
    si["widgets_values"] = siw

    # Prompts
    if positive is not None:
        pos = must_node(3)
        pv = pos.get("widgets_values", [""])
        pv[0] = positive.strip()
        pos["widgets_values"] = pv
    if negative is not None:
        neg = must_node(4)
        nv = neg.get("widgets_values", [""])
        nv[0] = negative.strip()
        neg["widgets_values"] = nv

    return wf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["style", "char"], default="style")
    ap.add_argument("--raw-set", required=True, help="Folder name under datasets/raw/{style|characters}/")
    ap.add_argument("--run-id", default=_now_run_id())
    ap.add_argument("--weights", default="0.4,0.6,0.8", help="Comma-separated LoRA weights to sweep")
    ap.add_argument("--seeds", default="123456789", help="Comma-separated seeds for bench")
    ap.add_argument("--bench-workflow", required=True, help="Path to bench workflow JSON")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--positive", default=None, help="Override positive prompt (optional)")
    ap.add_argument("--negative", default=None, help="Override negative prompt (optional)")
    ap.add_argument("--fix-owner", action="store_true", help="Run scripts/fix_datasets_owner.bash before training")
    ap.add_argument("--resize", action="store_true", help="Run scripts/resize_dataset.py before training")
    ap.add_argument("--yes", action="store_true", help="Pass -y to resize_dataset.py (overwrite outputs without prompt)")
    ap.add_argument("--lora-install-dir", default="models/loras", help="Where to copy trained LoRA so ComfyUI can load it")
    ap.add_argument("--run-lora-bash", default="scripts/run_lora.bash")
    ap.add_argument("--resize-script", default="scripts/resize_dataset.py")
    ap.add_argument("--fix-owner-bash", default="scripts/fix_datasets_owner.bash")
    ap.add_argument("--batch-size", type=int, default=1, help="Bench batch size (recommended: 1)")
    ap.add_argument("--skip-train", action="store_true", help="Skip kohya training (for rerun of bench)")
    ap.add_argument("--skip-bench", action="store_true", help="Skip ComfyUI bench generation (only preprocess/train)")
    args = ap.parse_args()

    run_id = args.run_id
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 0) Optional ownership fix
    if args.fix_owner:
        fix_script = REPO_ROOT / args.fix_owner_bash
        if fix_script.exists():
            run_cmd(["bash", str(fix_script)], cwd=REPO_ROOT)
        else:
            print(f"[warn] fix-owner script not found: {fix_script}")

    # 1) Optional resize / preprocess
    preprocess_log = REPO_ROOT / "datasets" / "normalized" / "preprocess_log.csv"
    if args.resize:
        resize_script = REPO_ROOT / args.resize_script
        cmd = ["python", str(resize_script), "--convert-rgb"]
        if args.yes:
            cmd.append("-y")
        # Important: run on datasets/raw so top-level remapping works (style/<set> -> style/<n_set>)
        cmd += ["--input", str(REPO_ROOT / "datasets" / "raw"), "--output", str(REPO_ROOT / "datasets" / "normalized")]
        # Save rejected to run folder for inspection
        rej_dir = run_dir / "rejected"
        cmd += ["--rejected", str(rej_dir)]
        run_cmd(cmd, cwd=REPO_ROOT)

        # snapshot preprocess log to run dir
        if preprocess_log.exists():
            shutil.copy2(preprocess_log, run_dir / "preprocess_log.csv")

    # 2) Resolve normalized set name and train
    group = "style" if args.profile == "style" else "characters"
    normalized_set = parse_normalized_set_name(preprocess_log, args.raw_set, group=group)
    base_name = base_name_from_set(normalized_set)

    # Expected output file name from container logic:
    #   style -> style_<base_name>
    #   char  -> char_<base_name>
    prefix = "style_" if args.profile == "style" else "char_"
    produced_name = f"{prefix}{base_name}.safetensors"
    trained_lora_host = REPO_ROOT / "output" / "kohya" / produced_name

    if not args.skip_train:
        run_lora = REPO_ROOT / args.run_lora_bash
        run_cmd(["bash", str(run_lora), "--profile", args.profile, "--input", normalized_set], cwd=REPO_ROOT)

        if not trained_lora_host.exists():
            print(f"[warn] trained LoRA not found at expected path: {trained_lora_host}")
            print("       Check output_dir in TOML (configs/lora/*.toml) and host volume mappings.")
        else:
            # snapshot into run dir with run_id suffix (avoid later overwrite)
            train_dir = run_dir / "train"
            train_dir.mkdir(exist_ok=True)
            archived = train_dir / f"{prefix}{base_name}__{run_id}.safetensors"
            shutil.copy2(trained_lora_host, archived)

            # install for ComfyUI using archived filename (so multiple versions can coexist)
            install_dir = REPO_ROOT / args.lora_install_dir
            install_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archived, install_dir / archived.name)

            # We'll use the archived filename for bench generation
            trained_lora_filename = archived.name
    else:
        # If skipping train, assume latest exists and install dir already has it;
        # fall back to the non-archived name.
        trained_lora_filename = trained_lora_host.name

    if args.skip_bench:
        print("[done] preprocess/train completed (bench skipped)")
        return 0

    # 3-5) Bench generation: weight sweep + metrics + contact sheets
    # Ensure ComfyUI can load the file we reference
    install_dir = REPO_ROOT / args.lora_install_dir
    if not (install_dir / trained_lora_filename).exists():
        if trained_lora_host.exists():
            shutil.copy2(trained_lora_host, install_dir / trained_lora_filename)
        else:
            print(f"[error] LoRA file missing both in install dir and output: {trained_lora_filename}")
            return 2

    bench_workflow_path = (REPO_ROOT / args.bench_workflow) if not Path(args.bench_workflow).is_absolute() else Path(args.bench_workflow)
    workflow = json.loads(bench_workflow_path.read_text(encoding="utf-8"))

    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    comfy = ComfyClient(base_url=args.comfy_url)

    produced_paths: List[Path] = []
    for w in weights:
        for s in seeds:
            # ComfyUI SaveImage writes under its output dir; subfolders are allowed in prefix.
            prefix_path = f"runs/{run_id}/bench/w{w:.2f}/seed{s}"
            wf2 = patch_workflow_for_bench(
                workflow,
                lora_filename=trained_lora_filename,
                lora_weight=w,
                seed=s,
                filename_prefix=prefix_path,
                positive=args.positive,
                negative=args.negative,
                batch_size=args.batch_size,
            )
            res = comfy.run_workflow_and_collect_images(wf2, timeout_s=900.0)

            # History returns filename/subfolder/type. Build host paths under repo/output
            for img in res.output_images:
                sub = img.get("subfolder", "")
                fn = img.get("filename", "")
                p = REPO_ROOT / "output" / sub / fn if sub else (REPO_ROOT / "output" / fn)
                produced_paths.append(p)

    # 5) Metrics + contact sheets
    img_metrics = []
    for p in produced_paths:
        if p.exists():
            try:
                img_metrics.append(compute_metrics(p))
            except Exception as e:
                print(f"[warn] metrics failed for {p}: {e}")
        else:
            print(f"[warn] output image not found (check Comfy output mapping): {p}")

    metrics_csv = run_dir / "bench" / "metrics.csv"
    write_metrics_csv(img_metrics, metrics_csv)

    # Create per-seed contact sheets (weights side-by-side)
    by_seed: dict[int, List[tuple[float, Path]]] = {}
    for m in img_metrics:
        ms = re.search(r"/w(\d+\.\d+)/seed(\d+)", str(m.path).replace("\\\\", "/"))
        if not ms:
            continue
        w_str, s_str = ms.group(1), ms.group(2)
        by_seed.setdefault(int(s_str), []).append((float(w_str), m.path))

    for seed, items in by_seed.items():
        items.sort(key=lambda t: t[0])
        paths = [p for _w, p in items]
        out = run_dir / "bench" / "grid" / f"seed{seed}_weights.png"
        try:
            make_contact_sheet(paths, out_path=out, cols=len(weights))
        except Exception as e:
            print(f"[warn] contact sheet failed for seed={seed}: {e}")

    print(f"[done] Run complete: {run_id}")
    print(f" - Run dir: {run_dir}")
    print(f" - Metrics: {metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
