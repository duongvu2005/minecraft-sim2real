"""Class scheme and the block-name rules that assign each Minecraft block a class."""

from __future__ import annotations

ROAD = 0
STRUCTURE = 1
VEGETATION = 2
SKY = 3
IGNORE = 255

CLASS_NAMES: tuple[str, ...] = ("road", "structure", "vegetation", "sky")

# Colors painted into the resource pack. Sky has no entry: Minecraft renders it
# procedurally and it is recovered in post-processing instead.
PACK_COLORS: dict[int, tuple[int, int, int]] = {
    ROAD: (255, 0, 0),
    STRUCTURE: (0, 0, 255),
    VEGETATION: (0, 255, 0),
    IGNORE: (0, 0, 0),
}

# Blocks Arnis lays down for road surfaces.
ROAD_BLOCKS: frozenset[str] = frozenset({
    "cyan_terracotta",
    "gray_concrete_powder",
})

VEGETATION_PATTERNS: tuple[str, ...] = (
    "leaves", "_log", "_wood", "stripped_", "sapling", "grass_block", "grass",
    "fern", "flower", "tulip", "rose", "dandelion", "poppy", "azalea", "vine",
    "moss", "lily_pad", "bush", "bamboo", "cactus", "sugar_cane", "mushroom",
    "wheat", "carrot", "potato", "beetroot", "berry", "kelp", "seagrass",
    "lichen", "spore_blossom", "pitcher_plant", "torchflower", "blue_orchid",
    "azure_bluet",
)

# Non-physical, animated or otherwise unlabelable blocks.
IGNORE_PATTERNS: tuple[str, ...] = (
    "water", "lava", "fire", "soul_fire", "portal", "command", "structure_",
    "barrier", "_spawner", "debug", "redstone", "piston", "rail",
    "destroy_stage",
)


def classify_block(texture_name: str) -> int:
    """Class index for a block texture name. Anything unmatched defaults to structure."""
    name = texture_name.lower()

    for pattern in IGNORE_PATTERNS:
        if pattern in name:
            return IGNORE

    if name in ROAD_BLOCKS or any(name.startswith(b) for b in ROAD_BLOCKS):
        return ROAD

    for pattern in VEGETATION_PATTERNS:
        if pattern in name:
            return VEGETATION

    return STRUCTURE
