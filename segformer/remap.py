"""Label maps from Cityscapes and ADE20K into the four-class scheme."""

from __future__ import annotations

import numpy as np

from pipeline.blocks import IGNORE, ROAD, SKY, STRUCTURE, VEGETATION

# Cityscapes raw label ids, as stored in labelIds.png. Anything not listed here
# is ignored.
CITYSCAPES: dict[int, tuple[str, int]] = {
    7: ("road", ROAD),
    8: ("sidewalk", STRUCTURE),
    11: ("building", STRUCTURE),
    12: ("wall", STRUCTURE),
    13: ("fence", STRUCTURE),
    21: ("vegetation", VEGETATION),
    22: ("terrain", IGNORE),
    23: ("sky", SKY),
}

# ADE20K classes as the model emits them, 0-indexed where the ADE docs are
# 1-indexed. Anything not listed here is ignored.
ADE20K: dict[int, tuple[str, int]] = {
    0: ("wall", STRUCTURE),
    1: ("building", STRUCTURE),
    2: ("sky", SKY),
    4: ("tree", VEGETATION),
    6: ("road", ROAD),
    8: ("sidewalk", STRUCTURE),
    13: ("grass", VEGETATION),
    17: ("plant", VEGETATION),
    25: ("house", STRUCTURE),
    29: ("field", VEGETATION),
    46: ("palm tree", VEGETATION),
    53: ("stairs", STRUCTURE),
    59: ("stairway", STRUCTURE),
    84: ("fence", STRUCTURE),
    90: ("bridge", STRUCTURE),
    96: ("bannister", STRUCTURE),
    100: ("runway", ROAD),
}

N_ADE_CLASSES = 150


def _lut(mapping: dict[int, tuple[str, int]], size: int) -> np.ndarray:
    lut = np.full(size, IGNORE, dtype=np.uint8)
    for index, (_, target) in mapping.items():
        lut[index] = target
    return lut


def cityscapes_lut() -> np.ndarray:
    """256 entries so it can index a raw labelIds image directly."""
    return _lut(CITYSCAPES, 256)


def ade20k_lut() -> np.ndarray:
    """150 entries, indexed by the baseline model's argmax."""
    return _lut(ADE20K, N_ADE_CLASSES)
