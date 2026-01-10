#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SUPPORTED_EXTS = {".webp", ".jpg", ".jpeg", ".png"}


def convert_image(src: Path, dst: Path) -> None:
    """
    Convert image to PNG, preserving alpha if exists.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        # Convert to RGBA to safely preserve transparency
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA")
        elif im.mode == "RGB":
            # keep RGB as-is
            pass

        im.save(dst, format="PNG", optimize=True)


def convert_one(
    src_str: str,
    dst_str: str,
    overwrite: bool,
    verify: bool,
) -> tuple[str, str | None, bool, str]:
    src = Path(src_str)
    dst = Path(dst_str)

    if dst.exists() and not overwrite:
        return "skipped", None, False, dst_str

    try:
        convert_image(src, dst)
        mismatch = False
        if verify:
            with Image.open(src) as im_src, Image.open(dst) as im_dst:
                mismatch = im_src.size != im_dst.size
        return "converted", None, mismatch, dst_str
    except Exception as e:
        return "failed", f"{src} ({e})", False, dst_str


def _convert_one_star(
    args: tuple[str, str, bool, bool],
) -> tuple[str, str | None, bool, str]:
    return convert_one(*args)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert datasets/raw images to PNG keeping directory structure"
    )
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_input = project_root / "datasets" / "raw"
    default_output = project_root / "datasets" / "raw_format"

    ap.add_argument(
        "--input",
        default=str(default_input),
        help="Input root directory (default: <project_root>/datasets/raw)",
    )
    ap.add_argument(
        "--output",
        default=str(default_output),
        help="Output root directory (default: <project_root>/datasets/raw_format)",
    )
    ap.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Overwrite existing files without confirmation",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=24,
        help="Number of worker processes (default: 24)",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Verify output size matches input size",
    )

    args = ap.parse_args()

    src_root = Path(args.input).resolve()
    dst_root = Path(args.output).resolve()

    if not src_root.exists():
        print(f"[error] input directory not found: {src_root}", file=sys.stderr)
        return 1

    sources = []
    for src in src_root.rglob("*"):
        if src.is_file() and src.suffix.lower() in SUPPORTED_EXTS:
            rel = src.relative_to(src_root)
            dst = (dst_root / rel).with_suffix(".png")
            sources.append((src, dst))

    converted = 0
    skipped = 0
    failed = 0
    mismatches = []

    if args.jobs < 1:
        print("[error] --jobs must be >= 1", file=sys.stderr)
        return 2

    if args.jobs > 1 and not args.yes:
        print(
            "[error] --jobs > 1 requires --yes (no interactive prompts in parallel)",
            file=sys.stderr,
        )
        return 2

    if args.jobs == 1:
        for src, dst in tqdm(sources, desc="converting", unit="img"):
            if dst.exists() and not args.yes:
                ans = input(f"[overwrite?] {dst} (y/N): ").strip().lower()
                if ans not in ("y", "yes"):
                    skipped += 1
                    continue

            status, msg, mismatch, dst_str = convert_one(
                str(src), str(dst), overwrite=True, verify=args.verify
            )
            if status == "converted":
                converted += 1
                if mismatch:
                    mismatches.append(dst_str)
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                if msg:
                    print(f"[warn] failed to convert: {msg}", file=sys.stderr)
    else:
        max_workers = min(args.jobs, os.cpu_count() or args.jobs)
        tasks = [(str(src), str(dst), args.yes, args.verify) for src, dst in sources]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for status, msg, mismatch, dst_str in tqdm(
                ex.map(_convert_one_star, tasks),
                total=len(tasks),
                desc="converting",
                unit="img",
            ):
                if status == "converted":
                    converted += 1
                    if mismatch:
                        mismatches.append(dst_str)
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    if msg:
                        print(f"[warn] failed to convert: {msg}", file=sys.stderr)

    print("---- done ----")
    print(f"converted: {converted}")
    print(f"skipped  : {skipped}")
    print(f"failed   : {failed}")
    print(f"output   : {dst_root}")
    if args.verify:
        print(f"size_mismatch: {len(mismatches)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
