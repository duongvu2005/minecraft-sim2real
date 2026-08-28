"""Convert the raw color masks Minecraft renders under the pack into integer labels."""

from __future__ import annotations

import numpy as np

from .blocks import IGNORE, ROAD, SKY, STRUCTURE, VEGETATION

# Colors for the human-readable check images. Ignore is white so it stands out
# against the black it has in the raw mask.
CHECK_COLORS: dict[int, tuple[int, int, int]] = {
    ROAD: (255, 0, 0),
    STRUCTURE: (0, 0, 255),
    VEGETATION: (0, 255, 0),
    SKY: (0, 255, 255),
    IGNORE: (255, 255, 255),
}


def color_mask_to_labels(rgb_mask: np.ndarray) -> np.ndarray:
    """
    Per-pixel channel dominance.

    In-game lighting darkens the pack's flat colors, so a structure pixel is
    usually some darker blue rather than (0, 0, 255). Road, structure and
    vegetation each keep one dominant channel under that darkening. Sky is the
    only thing with all three channels bright at once, so it is tested last and
    overrides the others.
    """
    R = rgb_mask[..., 0].astype(np.int16)
    G = rgb_mask[..., 1].astype(np.int16)
    B = rgb_mask[..., 2].astype(np.int16)

    labels = np.full(rgb_mask.shape[:2], IGNORE, dtype=np.uint8)

    is_road = (R > G + 40) & (R > B + 40) & (R > 50)
    is_vegetation = (G > R + 40) & (G > B + 40) & (G > 50)
    is_structure = (B > R + 40) & (B > G + 40) & (B > 50)
    is_sky = (R > 40) & (G > 60) & (B > 100)

    labels[is_road] = ROAD
    labels[is_vegetation] = VEGETATION
    labels[is_structure] = STRUCTURE
    labels[is_sky] = SKY

    return labels


def labels_to_check_image(labels: np.ndarray) -> np.ndarray:
    """Label map back to flat colors, for eyeballing a mask against its RGB frame."""
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for cls, color in CHECK_COLORS.items():
        out[labels == cls] = color
    return out
