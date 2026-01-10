#!/usr/bin/env python3
# Python 3.10.12
"""
Dataset preprocessor for LoRA training.
- Reads images from input_dir (recursively)
- Rejects images with extreme aspect ratios or too small dimensions
- Resizes while keeping aspect ratio
- Outputs sizes as multiples of 64 (recommended for SD training)
- Saves results to output_dir (mirrors folder structure with optional renaming)
- Writes a CSV log for rejected images

NEW:
- If input is under datasets/raw/{characters|style}/<set_name>/... and <set_name> does NOT start with digits,
  output folder name becomes "<N>_<set_name>" where N is assigned uniquely.
- N is chosen from 1..127 first. If exhausted, warn and continue with 128+.
- If a folder already starts with digits, keep it as-is.
  If its numeric prefix collides with another folder, warn and reassign for the later one.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Only these top-level groups are treated as "datasets groups"
GROUPS = {"characters", "style"}

PREFIXED_NAME_RE = re.compile(r"^(\d+)_+(.*)$")

# Extract leading integer prefix
LEADING_INT_RE = re.compile(r"^(\d+)")


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def round_to_multiple(x: int, m: int) -> int:
    return max(m, int(round(x / m) * m))


def compute_target_size(
    w: int,
    h: int,
    target_long: int,
    multiple: int,
) -> tuple[int, int]:
    """
    Resize to make long side ~= target_long while keeping aspect ratio,
    then round both sides to nearest 'multiple' (64 by default).
    """
    if w <= 0 or h <= 0:
        raise ValueError("Invalid dimensions")

    long_side = max(w, h)
    scale = target_long / float(long_side)

    w1 = max(1, int(round(w * scale)))
    h1 = max(1, int(round(h * scale)))

    w2 = round_to_multiple(w1, multiple)
    h2 = round_to_multiple(h1, multiple)

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
) -> tuple[bool, str]:
    """
    Returns (ok, reason). If ok=True, saved to dst. If False, not saved.
    """
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)

            if convert_rgb:
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
            resized = im.resize((tw, th), resample=Image.Resampling.LANCZOS)

            ensure_parent_dir(dst)

            ext = src.suffix.lower()
            if ext == ".png":
                out_path = dst.with_suffix(".png")
                resized.save(out_path, format="PNG", optimize=True)
            else:
                out_path = dst.with_suffix(".jpg")
                resized = resized.convert("RGB")
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


def process_one_task(
    args: tuple[str, str, str, int, int, int, float, bool],
) -> tuple[str, str, bool, str]:
    src_str, dst_str, rel2_str, target_long, multiple, min_short, max_ar, convert_rgb = (
        args
    )
    ok, reason = process_one(
        src=Path(src_str),
        dst=Path(dst_str),
        target_long=target_long,
        multiple=multiple,
        min_short=min_short,
        max_ar=max_ar,
        convert_rgb=convert_rgb,
    )
    return src_str, rel2_str, ok, reason


def iter_images(input_dir: Path) -> Iterable[Path]:
    for p in input_dir.rglob("*"):
        if is_image_file(p):
            yield p


def _parse_leading_int(name: str) -> int | None:
    m = LEADING_INT_RE.match(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _split_prefixed_name(name: str) -> tuple[int | None, str]:
    """
    "20_cyrene" -> (20, "cyrene")
    "cyrene"    -> (None, "cyrene")
    "20cyrene"  -> (20, "20cyrene")  # underscoreが無い場合はbaseとして扱う
    """
    m = PREFIXED_NAME_RE.match(name)
    if not m:
        # digits-only prefix (e.g. "20cyrene") is not treated as prefixed-set
        n = _parse_leading_int(name)
        if n is not None:
            return n, name
        return None, name

    try:
        n = int(m.group(1))
    except Exception:
        n = None
    base = m.group(2).strip() or name
    return n, base


def _gather_used_prefixes(output_dir: Path) -> dict[str, set[int]]:
    """
    Scan output_dir/{characters|style}/<dir> and collect numeric prefixes used.
    """
    used: dict[str, set[int]] = {g: set() for g in GROUPS}
    for g in GROUPS:
        group_path = output_dir / g
        if not group_path.exists():
            continue
        for child in group_path.iterdir():
            if not child.is_dir():
                continue
            n = _parse_leading_int(child.name)
            if n is not None:
                used[g].add(n)
    return used


def _gather_existing_sets(
    output_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, set[int]]]:
    """
    Returns:
      existing[group][base] = dir_name (e.g., existing["characters"]["cyrene"]="1_cyrene")
      used_prefixes[group] = {1,2,3,...}
    同じbaseが複数ある場合は、最小prefixのものを採用し警告。
    """
    existing: dict[str, dict[str, str]] = {g: {} for g in GROUPS}
    used: dict[str, set[int]] = {g: set() for g in GROUPS}

    for g in GROUPS:
        group_path = output_dir / g
        if not group_path.exists():
            continue

        for child in group_path.iterdir():
            if not child.is_dir():
                continue

            n, base = _split_prefixed_name(child.name)
            if n is not None:
                used[g].add(n)

            # base名の復元: "1_cyrene" の base は "cyrene"
            # ただし "20cyrene" みたいな名前は base=そのままなので、再利用対象としては弱いが害はない
            if base:
                if base in existing[g]:
                    # collision: 既に別のdir_nameが登録済み
                    prev = existing[g][base]
                    prev_n, _ = _split_prefixed_name(prev)
                    # より小さいprefixを優先
                    if prev_n is None or (n is not None and n < prev_n):
                        print(
                            f"[WARN] Multiple dirs for base '{g}/{base}': '{prev}' and '{child.name}'. Use '{child.name}'."
                        )
                        existing[g][base] = child.name
                    else:
                        print(
                            f"[WARN] Multiple dirs for base '{g}/{base}': '{prev}' and '{child.name}'. Use '{prev}'."
                        )
                else:
                    existing[g][base] = child.name

    return existing, used


def _assign_prefix(used: set[int]) -> int:
    """
    Pick an unused prefix. Prefer 1..127; if exhausted, warn and continue 128+.
    """
    for n in range(1, 128):
        if n not in used:
            used.add(n)
            return n

    # overflow (practical ok) -> keep going
    n = 128
    while n in used:
        n += 1
    print(f"[WARN] Prefix space 1..127 exhausted. Assigning {n} (beyond int8 range).")
    used.add(n)
    return n


def confirm_overwrite(path: Path, assume_yes: bool) -> bool:
    """
    Ask user if they want to overwrite an existing directory.
    If assume_yes=True, always overwrite without prompting.
    """
    if assume_yes:
        print(f"[OVERWRITE] (auto -y) {path}")
        return True

    while True:
        ans = (
            input(f"[CONFIRM] Output directory exists: {path}\nOverwrite? [y/N]: ")
            .strip()
            .lower()
        )
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            return False
        print("Please answer 'y' or 'n'.")


def build_topdir_mapping(
    input_dir: Path,
    output_dir: Path,
    assume_yes: bool,
) -> dict[tuple[str, str], str]:
    # 既存の base -> dir_name と、使用済みprefixを取得
    existing, used = _gather_existing_sets(output_dir)

    mapping: dict[tuple[str, str], str] = {}

    for g in GROUPS:
        group_path = input_dir / g
        if not group_path.exists():
            continue

        for d in sorted(
            [p for p in group_path.iterdir() if p.is_dir()], key=lambda p: p.name
        ):
            name = d.name  # raw側のセット名（例: "cyrene" or "20_cyrene"）
            n = _parse_leading_int(name)

            # 1) まず「rawの名前がdigits始まり」ならそのまま扱う
            if n is not None:
                out_name = name
                out_dir = output_dir / g / out_name

                # ただし prefix collision だけは回避
                if n in used[g] and not out_dir.exists():
                    new_n = _assign_prefix(used[g])
                    print(f"[WARN] Prefix collision in '{g}': '{name}' → {new_n}")
                    out_name = f"{new_n}_{name}"
                    out_dir = output_dir / g / out_name
                else:
                    used[g].add(n)

                # 上書き確認
                if out_dir.exists() and not confirm_overwrite(
                    out_dir, assume_yes=assume_yes
                ):
                    print(f"[SKIP] {g}/{name} skipped.")
                    continue

                mapping[(g, name)] = out_name
                continue

            # 2) rawの名前がdigits始まりでない場合：
            #    normalized側に "N_<name>" が既にあればそれを再利用（= 同一名称認識）
            if name in existing[g]:
                out_name = existing[g][name]
                out_dir = output_dir / g / out_name
                if out_dir.exists() and not confirm_overwrite(
                    out_dir, assume_yes=assume_yes
                ):
                    print(f"[SKIP] {g}/{name} skipped.")
                    continue

                mapping[(g, name)] = out_name
                continue

            # 3) 新規の場合のみ、未使用prefixを割当
            new_n = _assign_prefix(used[g])
            out_name = f"{new_n}_{name}"
            out_dir = output_dir / g / out_name

            if out_dir.exists() and not confirm_overwrite(
                out_dir, assume_yes=assume_yes
            ):
                print(f"[SKIP] {g}/{name} skipped.")
                continue

            mapping[(g, name)] = out_name
            # 新規割当した base を existing にも登録して、同一実行内での重複も防ぐ
            existing[g][name] = out_name

    return mapping


def remap_rel_path(rel: Path, mapping: dict[tuple[str, str], str]) -> Path:
    """
    If rel is like "{group}/{topdir}/...": remap topdir using mapping.
    Otherwise keep rel.
    """
    parts = rel.parts
    if len(parts) >= 2 and parts[0] in GROUPS:
        g = parts[0]
        top = parts[1]
        out_top = mapping.get((g, top), top)
        return Path(g) / out_top / Path(*parts[2:])
    return rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Overwrite existing output directories without prompting.",
    )
    ap.add_argument(
        "--input", type=Path, default=None, help="Input directory (images recursively)."
    )
    ap.add_argument("--output", type=Path, default=None, help="Output directory.")
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
        "--jobs",
        type=int,
        default=24,
        help="Number of worker processes (default: 24).",
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
        args.input if args.input is not None else (project_root / "datasets" / "raw")
    )
    output_dir = (
        args.output
        if args.output is not None
        else (project_root / "datasets" / "normalized")
    )
    rejected_dir = args.rejected

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if rejected_dir is not None:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    # NEW: build mapping for top-level set directories under raw/characters and raw/style
    mapping = build_topdir_mapping(
        input_dir=input_dir, output_dir=output_dir, assume_yes=args.yes
    )

    log_path = output_dir / "preprocess_log.csv"
    rows = []
    src_list = sorted(iter_images(input_dir))
    if not src_list:
        print(f"No images found in: {input_dir}")
        return 0

    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    tasks: list[tuple[str, str, str, int, int, int, float, bool]] = []
    for src in src_list:
        rel = src.relative_to(input_dir)
        rel2 = remap_rel_path(rel, mapping)
        dst = output_dir / rel2  # extension may be changed later
        tasks.append(
            (
                str(src),
                str(dst),
                str(rel2),
                args.target_long,
                args.multiple,
                args.min_short,
                args.max_ar,
                args.convert_rgb,
            )
        )

    if args.jobs == 1:
        it = (process_one_task(t) for t in tasks)
    else:
        max_workers = min(args.jobs, os.cpu_count() or args.jobs)
        ex = ProcessPoolExecutor(max_workers=max_workers)
        it = ex.map(process_one_task, tasks)

    try:
        for src_str, rel2_str, ok, reason in tqdm(
            it, total=len(tasks), desc="processing", unit="img"
        ):
            rows.append(
                {
                    "src": src_str,
                    "rel": rel2_str,
                    "ok": "1" if ok else "0",
                    "reason": reason,
                }
            )

            if (not ok) and rejected_dir is not None:
                rej_path = rejected_dir / rel2_str
                ensure_parent_dir(rej_path)
                try:
                    rej_path.write_bytes(Path(src_str).read_bytes())
                except Exception:
                    pass
    finally:
        if args.jobs > 1:
            ex.shutdown()

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
