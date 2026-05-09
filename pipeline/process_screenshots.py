"""
Pairs Minecraft F2 screenshots into RGB/mask pairs and converts masks
to integer label maps.

Workflow:
    1. In Minecraft: press F2 (RGB), swap pack, F2 (mask), swap back
    2. Repeat for all positions you want to capture
    3. Run this script to organize everything

Usage:
    python process_screenshots.py <screenshots_dir> <output_dataset_dir>

Example:
    python process_screenshots.py \\
        ~/Library/Application\\ Support/minecraft/screenshots \\
        ~/sim2real_dataset

Output structure:
    output_dataset_dir/
        rgb/           (renamed RGB screenshots)
            rgb_0000.png
            ...
        mask_color/    (raw RGB masks from Minecraft for reference)
            mask_0000.png
            ...
        mask_label/    (integer class labels - use these for training)
            mask_0000.png
            ...
        mask_check/    (pure-RGB visualization of labels for verification)
            mask_0000.png
            ...
"""

import sys
import shutil
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("Please install: pip install numpy Pillow")
    sys.exit(1)


# Class label scheme
CLASS_ROAD       = 0
CLASS_BUILDING   = 1
CLASS_VEGETATION = 2
CLASS_SKY        = 3
CLASS_IGNORE     = 255


def color_mask_to_labels(rgb_mask: np.ndarray) -> np.ndarray:
    """
    Convert an RGB color mask to integer class labels.

    Resource pack colors:
        Road       -> pure red    (255, 0, 0)
        Building   -> pure blue   (0, 0, 255)
        Vegetation -> pure green  (0, 255, 0)
        Sky        -> Minecraft's procedural sky (light blue gradient, R+G+B all bright)
        Ignore     -> pure black  (0, 0, 0)

    Key insight: sky is the only thing where ALL THREE channels are reasonably bright
    (it's a light desaturated blue). Buildings/road/veg are pure single-color
    (one channel dominant, others near zero), even after Minecraft's lighting darkens them.
    """
    R = rgb_mask[..., 0].astype(np.int16)
    G = rgb_mask[..., 1].astype(np.int16)
    B = rgb_mask[..., 2].astype(np.int16)

    labels = np.full(rgb_mask.shape[:2], CLASS_IGNORE, dtype=np.uint8)

    # Sky FIRST: all channels reasonably bright (light blue gradient).
    # Even at the darkest part of the sky gradient, R is still > 50.
    # Pure building blue has R near 0.
    is_sky = (R > 40) & (G > 60) & (B > 100)

    # For non-sky pixels, classify by dominant channel:
    is_road       = (R > G + 40) & (R > B + 40) & (R > 50)
    is_vegetation = (G > R + 40) & (G > B + 40) & (G > 50)
    is_building   = (B > R + 40) & (B > G + 40) & (B > 50)

    # Apply in priority order. Sky LAST so sky-blue pixels override anything else.
    labels[is_road]       = CLASS_ROAD
    labels[is_vegetation] = CLASS_VEGETATION
    labels[is_building]   = CLASS_BUILDING
    labels[is_sky]        = CLASS_SKY

    return labels


def label_to_check_image(labels: np.ndarray) -> np.ndarray:
    """
    Convert label map back to pure RGB colors for visual verification.
    Uses pure primary colors that match the resource pack scheme,
    and WHITE for ignore so it stands out (vs black ignore in original mask).
    """
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    color[labels == CLASS_ROAD]       = (255, 0, 0)      # pure red
    color[labels == CLASS_BUILDING]   = (0, 0, 255)      # pure blue
    color[labels == CLASS_VEGETATION] = (0, 255, 0)      # pure green
    color[labels == CLASS_SKY]        = (0, 255, 255)    # pure cyan
    color[labels == CLASS_IGNORE]     = (255, 255, 255)  # white (stands out)
    return color


def process_screenshots(screenshots_dir: Path, output_dir: Path, copy: bool = True):
    """Pair screenshots and process them into a dataset."""

    screenshots_dir = screenshots_dir.expanduser()
    output_dir = output_dir.expanduser()

    if not screenshots_dir.exists():
        print(f"ERROR: {screenshots_dir} does not exist")
        sys.exit(1)

    # Find all PNG screenshots, sorted by filename (which equals timestamp order)
    screenshots = sorted(screenshots_dir.glob("*.png"))
    if len(screenshots) == 0:
        print(f"No screenshots found in {screenshots_dir}")
        sys.exit(1)
    if len(screenshots) % 2 != 0:
        print(f"WARNING: Found {len(screenshots)} screenshots (odd number).")
        print("Last one will be ignored. Make sure pairs are RGB-then-mask.")

    n_pairs = len(screenshots) // 2
    print(f"Found {len(screenshots)} screenshots → {n_pairs} pairs.\n")

    # Create output structure
    rgb_dir = output_dir / "rgb"
    mask_color_dir = output_dir / "mask_color"
    mask_label_dir = output_dir / "mask_label"
    mask_check_dir = output_dir / "mask_check"
    for d in (rgb_dir, mask_color_dir, mask_label_dir, mask_check_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(n_pairs):
        rgb_src = screenshots[2 * i]
        mask_src = screenshots[2 * i + 1]

        idx = f"{i:04d}"

        # Copy RGB
        rgb_dst = rgb_dir / f"rgb_{idx}.png"
        if copy:
            shutil.copy(rgb_src, rgb_dst)
        else:
            shutil.move(str(rgb_src), str(rgb_dst))

        # Copy color mask
        mask_color_dst = mask_color_dir / f"mask_{idx}.png"
        if copy:
            shutil.copy(mask_src, mask_color_dst)
        else:
            shutil.move(str(mask_src), str(mask_color_dst))

        # Convert color mask -> integer label mask
        mask_img = np.array(Image.open(mask_color_dst).convert("RGB"))
        labels = color_mask_to_labels(mask_img)
        Image.fromarray(labels, mode="L").save(mask_label_dir / f"mask_{idx}.png")

        # Also save a visual check version with pure RGB colors (white for ignore)
        check_img = label_to_check_image(labels)
        Image.fromarray(check_img).save(mask_check_dir / f"mask_{idx}.png")

        if (i + 1) % 50 == 0 or i == n_pairs - 1:
            print(f"  Processed {i + 1}/{n_pairs} pairs")

    # Print class distribution stats
    print("\nComputing class distribution...")
    total_pixels = 0
    class_pixels = {0: 0, 1: 0, 2: 0, 3: 0, 255: 0}
    for label_path in mask_label_dir.glob("*.png"):
        labels = np.array(Image.open(label_path))
        total_pixels += labels.size
        for cls in class_pixels:
            class_pixels[cls] += int((labels == cls).sum())

    print(f"\n{'Class':<14} {'Pixels':<14} {'Percentage':<10}")
    print(f"{'-'*40}")
    names = {0: "road", 1: "building", 2: "vegetation", 3: "sky", 255: "ignore"}
    for cls, count in class_pixels.items():
        pct = 100 * count / total_pixels if total_pixels else 0
        print(f"{names[cls]:<14} {count:<14,} {pct:<6.2f}%")

    print(f"\nDataset created at: {output_dir.absolute()}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    screenshots_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    process_screenshots(screenshots_dir, output_dir, copy=True)


if __name__ == "__main__":
    main()