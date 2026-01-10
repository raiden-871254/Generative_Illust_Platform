#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image, ImageOps, ImageStat, ImageFilter


@dataclass
class ImageMetrics:
    path: Path
    mean_luma: float
    std_luma: float
    laplacian_var: float  # sharpness-ish
    size_w: int
    size_h: int


def compute_metrics(img_path: Path) -> ImageMetrics:
    im = Image.open(img_path).convert("RGB")
    w, h = im.size

    # Luma stats
    gray = ImageOps.grayscale(im)
    stat = ImageStat.Stat(gray)
    mean = float(stat.mean[0])
    std = float(stat.stddev[0])

    # Laplacian variance (simple proxy for noise/detail)
    # PIL doesn't have laplacian directly, approximate with FIND_EDGES then variance.
    edges = gray.filter(ImageFilter.FIND_EDGES)
    est = ImageStat.Stat(edges)
    lap_var = float(est.var[0])

    return ImageMetrics(
        path=img_path,
        mean_luma=mean,
        std_luma=std,
        laplacian_var=lap_var,
        size_w=w,
        size_h=h,
    )


def write_metrics_csv(metrics: List[ImageMetrics], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "mean_luma", "std_luma", "laplacian_var", "w", "h"])
        for m in metrics:
            w.writerow([str(m.path), f"{m.mean_luma:.3f}", f"{m.std_luma:.3f}", f"{m.laplacian_var:.3f}", m.size_w, m.size_h])


def make_contact_sheet(
    image_paths: List[Path],
    out_path: Path,
    thumb_w: int = 384,
    padding: int = 8,
    cols: int = 3,
) -> None:
    """
    Simple grid for quick human selection.
    Keeps aspect ratio; centers thumbnails.
    """
    if not image_paths:
        raise ValueError("No images for contact sheet")

    thumbs: List[Image.Image] = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb_w, thumb_w * 10), Image.Resampling.LANCZOS)
        thumbs.append(im)

    rows = math.ceil(len(thumbs) / cols)
    max_h = max(t.size[1] for t in thumbs)
    cell_w = thumb_w
    cell_h = max_h

    sheet_w = cols * cell_w + (cols + 1) * padding
    sheet_h = rows * cell_h + (rows + 1) * padding
    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))

    for idx, im in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x0 = padding + c * (cell_w + padding)
        y0 = padding + r * (cell_h + padding)
        # center
        x = x0 + (cell_w - im.size[0]) // 2
        y = y0 + (cell_h - im.size[1]) // 2
        sheet.paste(im, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
