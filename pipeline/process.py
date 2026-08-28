"""Pair captured screenshots into (rgb, mask) files and write the label maps."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .blocks import CLASS_NAMES, IGNORE
from .labels import color_mask_to_labels, labels_to_check_image

SUBDIRS = ("rgb", "mask_color", "mask_label", "mask_check")


def pair_screenshots(screenshots_dir: Path) -> list[tuple[Path, Path]]:
    """
    Screenshots in capture order, taken two at a time as (rgb, mask).

    Minecraft names screenshots by timestamp, so sorting by filename recovers the
    order they were taken in, and capture alternates RGB then mask.
    """
    shots = sorted(screenshots_dir.glob("*.png"))
    if not shots:
        raise SystemExit(f"No screenshots found in {screenshots_dir}")
    if len(shots) % 2:
        print(f"WARNING: {len(shots)} screenshots is odd; ignoring the last one.")
    return [(shots[2 * i], shots[2 * i + 1]) for i in range(len(shots) // 2)]


def build_dataset(screenshots_dir: Path, output_dir: Path, move: bool = False) -> int:
    """Write rgb/, mask_color/, mask_label/ and mask_check/ under output_dir."""
    pairs = pair_screenshots(screenshots_dir)
    dirs = {name: output_dir / name for name in SUBDIRS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    transfer = shutil.move if move else shutil.copy
    for i, (rgb_src, mask_src) in enumerate(pairs):
        idx = f"{i:04d}"
        mask_color = dirs["mask_color"] / f"mask_{idx}.png"
        transfer(str(rgb_src), str(dirs["rgb"] / f"rgb_{idx}.png"))
        transfer(str(mask_src), str(mask_color))

        labels = color_mask_to_labels(np.array(Image.open(mask_color).convert("RGB")))
        Image.fromarray(labels, mode="L").save(dirs["mask_label"] / f"mask_{idx}.png")
        Image.fromarray(labels_to_check_image(labels)).save(
            dirs["mask_check"] / f"mask_{idx}.png"
        )

        if (i + 1) % 50 == 0 or i == len(pairs) - 1:
            print(f"  processed {i + 1}/{len(pairs)} pairs")

    return len(pairs)


def class_distribution(mask_label_dir: Path) -> tuple[Counter, int]:
    """Pixel counts per class over every label map in a directory."""
    counts: Counter = Counter()
    total = 0
    for path in sorted(mask_label_dir.glob("*.png")):
        labels = np.array(Image.open(path))
        counts.update(dict(zip(*np.unique(labels, return_counts=True))))
        total += labels.size
    return counts, total


def print_class_distribution(counts: Counter, total: int) -> None:
    label = {**{i: n for i, n in enumerate(CLASS_NAMES)}, IGNORE: "ignore"}
    print(f"\n{'class':<14}{'pixels':>16}{'percent':>10}")
    print("-" * 40)
    for cls in [0, 1, 2, 3, IGNORE]:
        n = int(counts.get(cls, 0))
        print(f"{label[cls]:<14}{n:>16,}{100 * n / total if total else 0:>9.2f}%")
