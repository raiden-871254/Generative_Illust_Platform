#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from bench_utils import compute_metrics, make_contact_sheet, write_metrics_csv
from comfy_client import ComfyClient
from comfy_docker import ComfyLogsTail, docker_compose, wait_for_comfy

REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def docker_compose(args: list[str], compose_file: Path, cwd: Path) -> None:
    cmd = ["docker", "compose", "-f", str(compose_file)] + args
    run_cmd(cmd, cwd=cwd)


def wait_for_comfy(url: str, timeout_s: int = 180) -> None:
    check_url = url.rstrip("/") + "/system_stats"
    start = time.time()
    last_err: str | None = None
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(check_url, timeout=3) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as e:
            last_err = str(e)
        time.sleep(1)
    raise TimeoutError(f"ComfyUI not ready in {timeout_s}s. last_err={last_err}")


def patch_workflow_for_bench(
    workflow: dict[str, Any],
    *,
    lora_filename: str,
    lora_weight: float,
    seed: int,
    filename_prefix: str,
    positive: str | None = None,
    negative: str | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    Works with ComfyUI API-format JSON like your bench file:
      - LoraLoader node id=11
      - KSampler node id=6
      - EmptyLatentImage node id=5
      - SaveImage node id=7
      - Positive/Negative prompt nodes id=3/4
    """
    wf = json.loads(json.dumps(workflow))  # deep copy

    def must_node(nid: int) -> dict[str, Any]:
        key = str(nid)
        if key not in wf:
            raise KeyError(f"Node id {nid} not found in workflow")
        return wf[key]

    # LoraLoader
    lora = must_node(11)
    inputs = lora.setdefault("inputs", {})
    inputs["lora_name"] = lora_filename
    inputs["strength_model"] = float(lora_weight)
    inputs["strength_clip"] = float(lora_weight)

    # KSampler seed
    ks = must_node(6)
    ks_inputs = ks.setdefault("inputs", {})
    ks_inputs["seed"] = int(seed)

    # EmptyLatentImage batch size
    el = must_node(5)
    el_inputs = el.setdefault("inputs", {})
    el_inputs["batch_size"] = int(batch_size)

    # SaveImage prefix (supports subfolders)
    si = must_node(7)
    si_inputs = si.setdefault("inputs", {})
    si_inputs["filename_prefix"] = filename_prefix

    # Prompts
    if positive is not None:
        pos = must_node(3)
        pos_inputs = pos.setdefault("inputs", {})
        pos_inputs["text"] = positive.strip()
    if negative is not None:
        neg = must_node(4)
        neg_inputs = neg.setdefault("inputs", {})
        neg_inputs["text"] = negative.strip()

    return wf


def _parse_seeds_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_weights_csv(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ComfyUI bench runner: stability check or harvest mode"
    )
    ap.add_argument("--mode", choices=["stability", "harvest"], required=True)
    ap.add_argument("--run-id", default=_now_run_id())

    # Comfy
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument(
        "--comfy-compose-file",
        default="comfy-docker/docker-compose.yml",
        help="comfy compose file",
    )
    ap.add_argument("--comfy-service", default="comfyui")
    ap.add_argument("--comfy-wait-timeout", type=int, default=180)
    ap.add_argument(
        "--start-comfy",
        action="store_true",
        default=True,
        help="Start comfyui before running (default: on)",
    )
    ap.add_argument("--no-start-comfy", action="store_true")
    ap.add_argument(
        "--stop-comfy-after",
        action="store_true",
        default=True,
        help="Stop comfyui after generation (default: on)",
    )
    ap.add_argument("--no-stop-comfy-after", action="store_true")

    # Workflow + prompts
    ap.add_argument(
        "--workflow",
        default="configs/gui/MeinaMix_v12_lora_bench.json",
        help="bench workflow JSON (API format)",
    )
    ap.add_argument("--positive", default=None)
    ap.add_argument("--negative", default=None)

    # LoRA
    ap.add_argument(
        "--lora-file",
        required=True,
        help="LoRA filename under models/loras (e.g. style_xxx__RUNID.safetensors)",
    )

    # Mode parameters
    ap.add_argument(
        "--weights",
        default="0.6",
        help="Comma-separated weights (stability: often single 0.6; harvest: usually 0.6)",
    )
    ap.add_argument(
        "--seeds",
        default="123456789,987654321",
        help="Comma-separated seeds for stability (ignored in harvest unless you set --harvest-seeds)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="stability: recommended 1; harvest: 8-32",
    )
    ap.add_argument(
        "--harvest-count",
        type=int,
        default=50,
        help="harvest: how many prompts to submit (each creates batch_size images)",
    )
    ap.add_argument(
        "--harvest-seed-min",
        type=int,
        default=1,
        help="harvest seed range min (inclusive)",
    )
    ap.add_argument(
        "--harvest-seed-max",
        type=int,
        default=2_000_000_000,
        help="harvest seed range max (inclusive)",
    )
    ap.add_argument(
        "--out-root",
        default="output/bench_runs",
        help="Output root under repo (default: output/bench_runs)",
    )
    args = ap.parse_args()

    if args.no_start_comfy:
        args.start_comfy = False
    if args.no_stop_comfy_after:
        args.stop_comfy_after = False

    compose_file = (
        (REPO_ROOT / args.comfy_compose_file)
        if not Path(args.comfy_compose_file).is_absolute()
        else Path(args.comfy_compose_file)
    ).resolve()
    comfy_cwd = compose_file.parent

    # Validate lora file exists
    lora_path = REPO_ROOT / "models" / "loras" / args.lora_file
    if not lora_path.exists():
        print(f"[error] LoRA not found: {lora_path}")
        return 2

    # Load workflow
    wf_path = (
        (REPO_ROOT / args.workflow)
        if not Path(args.workflow).is_absolute()
        else Path(args.workflow)
    )
    workflow = json.loads(wf_path.read_text(encoding="utf-8"))

    run_id = args.run_id
    out_root = REPO_ROOT / "output" / "bench" / "stability" / run_id
    (out_root / "grid").mkdir(parents=True, exist_ok=True)

    # Start comfy if requested
    if args.start_comfy:
        tail = ComfyLogsTail(
            compose_file=compose_file, cwd=comfy_cwd, service=args.comfy_service
        )

        docker_compose(
            ["up", "-d", args.comfy_service], compose_file=compose_file, cwd=comfy_cwd
        )
        tail.start()
        wait_for_comfy(args.comfy_url, timeout_s=args.comfy_wait_timeout)

    comfy = ComfyClient(base_url=args.comfy_url)

    weights = _parse_weights_csv(args.weights)
    produced_paths: list[Path] = []
    metrics_items = []

    def host_path_from_comfy_image(img: dict[str, Any]) -> Path:
        sub = img.get("subfolder", "")
        fn = img.get("filename", "")
        # Comfy SaveImage returns subfolder relative to its output root.
        # In your setup, that's typically repo/output.
        return (REPO_ROOT / "output" / sub / fn) if sub else (REPO_ROOT / "output" / fn)

    if args.mode == "stability":
        seeds = _parse_seeds_csv(args.seeds)
        for w in weights:
            for s in seeds:
                prefix_path = f"bench/stability/{run_id}/w{w:.2f}/seed{s}"
                wf2 = patch_workflow_for_bench(
                    workflow,
                    lora_filename=args.lora_file,
                    lora_weight=w,
                    seed=s,
                    filename_prefix=prefix_path,
                    positive=args.positive,
                    negative=args.negative,
                    batch_size=args.batch_size,
                )
                res = comfy.run_workflow_and_collect_images(wf2, timeout_s=900.0)
                for img in res.output_images:
                    p = host_path_from_comfy_image(img)
                    produced_paths.append(p)

    else:  # harvest
        # seed randomize per submission; each submission generates batch_size images.
        for _i in range(args.harvest_count):
            s = random.randint(args.harvest_seed_min, args.harvest_seed_max)
            for w in weights:
                prefix_path = (
                    f"{Path(args.out_root).name}/{args.mode}/{run_id}/w{w:.2f}/seed{s}"
                )
                wf2 = patch_workflow_for_bench(
                    workflow,
                    lora_filename=args.lora_file,
                    lora_weight=w,
                    seed=s,
                    filename_prefix=prefix_path,
                    positive=args.positive,
                    negative=args.negative,
                    batch_size=args.batch_size,
                )
                res = comfy.run_workflow_and_collect_images(wf2, timeout_s=900.0)
                for img in res.output_images:
                    p = host_path_from_comfy_image(img)
                    produced_paths.append(p)

    # Metrics
    for p in produced_paths:
        if p.exists():
            try:
                m = compute_metrics(p)
                metrics_items.append(m)
            except Exception as e:
                print(f"[warn] metrics failed for {p}: {e}")
        else:
            print(f"[warn] output image not found: {p}")

    metrics_csv = out_root / "metrics.csv"
    write_metrics_csv(metrics_items, metrics_csv)

    # Contact sheets: per-seed (weights side-by-side) if we can parse /wX.XX/seedNN
    by_seed: dict[int, list[tuple[float, Path]]] = {}
    for m in metrics_items:
        ms = re.search(r"/w(\d+\.\d+)/seed(\d+)", str(m.path).replace("\\", "/"))
        if not ms:
            continue
        w_str, s_str = ms.group(1), ms.group(2)
        by_seed.setdefault(int(s_str), []).append((float(w_str), m.path))

    for seed, items in by_seed.items():
        items.sort(key=lambda t: t[0])
        paths = [p for _w, p in items]
        out = out_root / "grid" / f"seed{seed}_weights.png"
        try:
            make_contact_sheet(paths, out_path=out, cols=len(weights))
        except Exception as e:
            print(f"[warn] contact sheet failed for seed={seed}: {e}")

    # Stop comfy if requested
    if args.stop_comfy_after:
        try:
            tail.stop()
            docker_compose(
                ["stop", args.comfy_service], compose_file=compose_file, cwd=comfy_cwd
            )
        except subprocess.CalledProcessError as e:
            print(f"[warn] failed to stop comfyui (ignored): {e}")

    print("[done]")
    print(f" - mode     : {args.mode}")
    print(f" - run_id   : {run_id}")
    print(f" - out_root : {out_root}")
    print(f" - metrics  : {metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
