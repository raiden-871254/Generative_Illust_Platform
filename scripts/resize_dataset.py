#!/usr/bin/env python3
# Python 3.10.12
"""
Dataset preprocessor for LoRA training.
- Reads images from input_dir (recursively)
- Rejects images with extreme aspect ratios or too small dimensions
- Resizes while keeping aspect ratio
- Outputs sizes as multiples of 64 (recommended for SD training)
- Saves results to output_dir (mirrors folder structure)
- Writes a CSV log for rejected images

Usage example:
  python tools/resize_dataset.py \
    --input ./input \
    --output ./output \
    --target-long 768 \
    --min-short 512 \
    --max-ar 2.2
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def round_to_multiple(x: int, m: int) -> int:
    return max(m, int(round(x / m) * m))


def clamp_min_multiple(x: int, m: int) -> int:
    # ensure >= m and multiple of m
    return max(m, (x // m) * m)


def compute_target_size(
    w: int,
    h: int,
    target_long: int,
    multiple: int,
) -> Tuple[int, int]:
    """
    Resize to make long side ~= target_long while keeping aspect ratio,
    then round both sides to nearest 'multiple' (64 by default).
    """
    if w <= 0 or h <= 0:
        raise ValueError("Invalid dimensions")

    long_side = max(w, h)
    scale = target_long / float(long_side)

    # First pass: float-scaled size
    w1 = max(1, int(round(w * scale)))
    h1 = max(1, int(round(h * scale)))

    # Round to multiples of 64 (slight distortion risk if rounding is large)
    w2 = round_to_multiple(w1, multiple)
    h2 = round_to_multiple(h1, multiple)

    # Ensure not zero
    w2 = max(multiple, w2)
    h2 = max(multiple, h2)

    return w2, h2


def aspect_ratio_ok(w: int, h: int, max_ar: float) -> bool:
    ar = w / float(h)
    return (1.0 / max_ar) <= ar <= max_ar


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def process_one(
    src: Path,
    dst: Path,
    target_long: int,
    multiple: int,
    min_short: int,
    max_ar: float,
    convert_rgb: bool,
) -> Tuple[bool, str]:
    """
    Returns (ok, reason). If ok=True, saved to dst. If False, not saved.
    """
    try:
        with Image.open(src) as im:
            # Fix orientation based on EXIF (important for phone images)
            im = ImageOps.exif_transpose(im)

            if convert_rgb:
                # Convert to RGB to avoid mode issues (e.g., P, LA)
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")

            w, h = im.size
            if min(w, h) < min_short:
                return False, f"too_small(min_side<{min_short})"

            if not aspect_ratio_ok(w, h, max_ar=max_ar):
                ar = w / float(h)
                return False, f"bad_aspect_ratio(ar={ar:.3f}, max_ar={max_ar})"

            tw, th = compute_target_size(
                w, h, target_long=target_long, multiple=multiple
            )

            # High-quality down/up sampling
            resized = im.resize((tw, th), resample=Image.Resampling.LANCZOS)

            ensure_parent_dir(dst)

            # Preserve PNG if input is PNG; otherwise save as JPG (smaller)
            # You can force a single format by editing here.
            ext = src.suffix.lower()
            if ext == ".png":
                out_path = dst.with_suffix(".png")
                resized.save(out_path, format="PNG", optimize=True)
            else:
                out_path = dst.with_suffix(".jpg")
                resized = resized.convert("RGB")  # JPEG needs RGB
                resized.save(
                    out_path,
                    format="JPEG",
                    quality=95,
                    optimize=True,
                    progressive=True,
                )

            return True, "ok"

    except Exception as e:
        return False, f"error({type(e).__name__}): {e}"


def iter_images(input_dir: Path) -> Iterable[Path]:
    for p in input_dir.rglob("*"):
        if is_image_file(p):
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input directory (images recursively).",
    )
    ap.add_argument(
        "--output", type=Path, default=None, help="Output directory."
    )
    ap.add_argument(
        "--target-long",
        type=int,
        default=768,
        help="Target long-side size before rounding (default: 768).",
    )
    ap.add_argument(
        "--multiple",
        type=int,
        default=64,
        help="Round output size to multiple (default: 64).",
    )
    ap.add_argument(
        "--min-short",
        type=int,
        default=512,
        help="Reject if min(w,h) < this (default: 512).",
    )
    ap.add_argument(
        "--max-ar",
        type=float,
        default=2.2,
        help="Reject if aspect ratio outside [1/max_ar, max_ar].",
    )
    ap.add_argument(
        "--convert-rgb",
        action="store_true",
        help="Convert images to RGB/RGBA friendly formats (recommended).",
    )
    ap.add_argument(
        "--rejected",
        type=Path,
        default=None,
        help="Optional directory to copy rejected items (keeps relative paths).",
    )
    args = ap.parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_dir = (
        args.input
        if args.input is not None
        else (project_root / "datasets" / "raw")
    )
    output_dir = (
        args.output
        if args.output is not None
        else (project_root / "normalized")
    )
    rejected_dir = args.rejected

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    # output は「無ければ作る」のは今の仕様でOK
    output_dir.mkdir(parents=True, exist_ok=True)

    # rejected を指定した場合は作る
    if rejected_dir is not None:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "preprocess_log.csv"

    rows = []
    src_list = sorted(iter_images(input_dir))
    if not src_list:
        print(f"No images found in: {input_dir}")
        return 0

    for src in tqdm(src_list, desc="processing", unit="img"):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel  # extension may be changed later

        ok, reason = process_one(
            src=src,
            dst=dst,
            target_long=args.target_long,
            multiple=args.multiple,
            min_short=args.min_short,
            max_ar=args.max_ar,
            convert_rgb=args.convert_rgb,
        )

        rows.append(
            {
                "src": str(src),
                "rel": str(rel),
                "ok": "1" if ok else "0",
                "reason": reason,
            }
        )

        if (not ok) and rejected_dir is not None:
            rej_path = rejected_dir / rel
            ensure_parent_dir(rej_path)
            # Copy as-is (do not convert)
            try:
                rej_path.write_bytes(src.read_bytes())
            except Exception:
                # ignore copy errors
                pass

    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["src", "rel", "ok", "reason"])
        w.writeheader()
        w.writerows(rows)

    ok_count = sum(1 for r in rows if r["ok"] == "1")
    ng_count = len(rows) - ok_count
    print(f"Done. OK={ok_count}, Rejected={ng_count}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
