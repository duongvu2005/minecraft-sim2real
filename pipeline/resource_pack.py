"""Build the flat-color resource pack that turns Minecraft's render into a mask."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from .blocks import CLASS_NAMES, IGNORE, PACK_COLORS, classify_block

PACK_FORMAT = 34


def block_texture_names(jar_path: Path) -> list[str]:
    """Every block texture name in a Minecraft client jar."""
    with zipfile.ZipFile(jar_path) as jar:
        names = [
            Path(n).stem
            for n in jar.namelist()
            if n.startswith("assets/minecraft/textures/block/") and n.endswith(".png")
        ]
    return sorted(set(names))


def _clear_output_dir(output_dir: Path) -> None:
    """Remove a previous pack, refusing to touch a directory we did not write."""
    if not output_dir.exists():
        return
    if not (output_dir / "pack.mcmeta").exists():
        raise SystemExit(
            f"{output_dir} exists and has no pack.mcmeta, so it is not a pack we "
            f"built. Refusing to delete it. Pass a different output directory."
        )
    shutil.rmtree(output_dir)


def build_pack(jar_path: Path, output_dir: Path) -> tuple[Counter, list[tuple[str, int]]]:
    """Write a 16x16 solid PNG per block, colored by its class."""
    if not jar_path.exists():
        raise SystemExit(f"Minecraft jar not found: {jar_path}")

    names = block_texture_names(jar_path)
    _clear_output_dir(output_dir)

    block_dir = output_dir / "assets" / "minecraft" / "textures" / "block"
    block_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pack.mcmeta").write_text(
        json.dumps(
            {"pack": {"pack_format": PACK_FORMAT,
                      "description": "Sim2Real Segmentation Pack (4 classes)"}},
            indent=2,
        )
    )

    counts: Counter = Counter()
    assignments: list[tuple[str, int]] = []
    for name in names:
        cls = classify_block(name)
        Image.new("RGB", (16, 16), PACK_COLORS[cls]).save(block_dir / f"{name}.png")
        counts[cls] += 1
        assignments.append((name, cls))

    return counts, assignments


def write_classification_log(path: Path, assignments: list[tuple[str, int]]) -> None:
    """One block per line, grouped by class."""
    order = [0, 2, 1, IGNORE]
    label = {**{i: n for i, n in enumerate(CLASS_NAMES)}, IGNORE: "ignore"}
    lines = ["Block texture -> Class", "=" * 60]
    for cls in order:
        members = [n for n, c in assignments if c == cls]
        lines.append(f"\n## {label[cls].upper()} ({len(members)} blocks)")
        lines.extend(f"  {n}" for n in members)
    path.write_text("\n".join(lines) + "\n")
