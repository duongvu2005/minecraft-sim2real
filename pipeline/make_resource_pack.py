"""
Generates a flat-color Minecraft resource pack for semantic segmentation.

Usage:
    python make_resource_pack.py <path_to_minecraft_jar> [output_dir]

Example:
    python make_resource_pack.py ~/Library/Application\\ Support/minecraft/versions/1.21.11/1.21.11.jar

The output is a folder you can drop into ~/.minecraft/resourcepacks/ (or zip it).

Class scheme (4 classes):
    0  road        -> RED      (255, 0, 0)
    1  building    -> BLUE     (0, 0, 255)   [DEFAULT - catches all wall/ground materials]
    2  vegetation  -> GREEN    (0, 255, 0)   [includes grass_block, leaves, logs, plants]
    3  sky         -> CYAN     (0, 255, 255) [handled via sky color, not blocks]
    255 ignore    -> BLACK    (0, 0, 0)
"""

import sys
import json
import zipfile
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Please install Pillow: pip install Pillow")
    sys.exit(1)


CLASS_COLORS = {
    "road":       (255, 0, 0),
    "building":   (0, 0, 255),
    "vegetation": (0, 255, 0),
    "ignore":     (0, 0, 0),
}

ROAD_BLOCKS = {
    "cyan_terracotta",
    "gray_concrete_powder",
}

VEGETATION_PATTERNS = (
    "leaves",
    "_log",
    "_wood",
    "stripped_",
    "sapling",
    "grass_block",
    "grass",
    "fern",
    "flower",
    "tulip",
    "rose",
    "dandelion",
    "poppy",
    "azalea",
    "vine",
    "moss",
    "lily_pad",
    "bush",
    "bamboo",
    "cactus",
    "sugar_cane",
    "mushroom",
    "wheat",
    "carrot",
    "potato",
    "beetroot",
    "berry",
    "kelp",
    "seagrass",
    "lichen",
    "spore_blossom",
    "pitcher_plant",
    "torchflower",
    "blue_orchid",
    "azure_bluet",
)

IGNORE_PATTERNS = (
    "water",
    "lava",
    "fire",
    "soul_fire",
    "portal",
    "command",
    "structure_",
    "barrier",
    "_spawner",
    "debug",
    "redstone",
    "piston",
    "rail",
    "destroy_stage",
)


def classify_block(texture_name: str) -> str:
    name = texture_name.lower()

    for pattern in IGNORE_PATTERNS:
        if pattern in name:
            return "ignore"

    if name in ROAD_BLOCKS or any(name.startswith(b) for b in ROAD_BLOCKS):
        return "road"

    for pattern in VEGETATION_PATTERNS:
        if pattern in name:
            return "vegetation"

    return "building"


def get_block_texture_names(jar_path: Path):
    if not jar_path.exists():
        print(f"ERROR: Minecraft jar not found at {jar_path}")
        sys.exit(1)

    texture_names = []
    with zipfile.ZipFile(jar_path, "r") as jar:
        for name in jar.namelist():
            if name.startswith("assets/minecraft/textures/block/") and name.endswith(".png"):
                fname = Path(name).stem
                texture_names.append(fname)

    return sorted(set(texture_names))


def make_solid_png(color, path: Path):
    img = Image.new("RGB", (16, 16), color)
    img.save(path)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    jar_path = Path(sys.argv[1]).expanduser()
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("segmentation_pack")

    print(f"Reading textures from: {jar_path}")
    texture_names = get_block_texture_names(jar_path)
    print(f"Found {len(texture_names)} block textures.\n")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    block_dir = output_dir / "assets" / "minecraft" / "textures" / "block"
    block_dir.mkdir(parents=True, exist_ok=True)

    mcmeta = {
        "pack": {
            "pack_format": 34,
            "description": "Sim2Real Segmentation Pack (4 classes)"
        }
    }
    with open(output_dir / "pack.mcmeta", "w") as f:
        json.dump(mcmeta, f, indent=2)

    class_counts = {cls: 0 for cls in CLASS_COLORS}
    classification_log = []

    for tex_name in texture_names:
        cls = classify_block(tex_name)
        color = CLASS_COLORS[cls]
        out_path = block_dir / f"{tex_name}.png"
        make_solid_png(color, out_path)
        class_counts[cls] += 1
        classification_log.append((tex_name, cls))

    log_path = output_dir.parent / "classification_log.txt"
    with open(log_path, "w") as f:
        f.write("Block texture -> Class\n")
        f.write("=" * 60 + "\n")
        for cls_name in ("road", "vegetation", "building", "ignore"):
            f.write(f"\n## {cls_name.upper()} ({class_counts[cls_name]} blocks)\n")
            for tex, cls in classification_log:
                if cls == cls_name:
                    f.write(f"  {tex}\n")

    print(f"Resource pack created at: {output_dir.absolute()}")
    print(f"\nClass distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls:<12} {count:>5} blocks")
    print(f"\nClassification log: {log_path.absolute()}")
    print(f"\nNext steps:")
    print(f"  1. Review {log_path.name} to check the classification")
    print(f"  2. Move '{output_dir.name}' to ~/Library/Application Support/minecraft/resourcepacks/")
    print(f"  3. Enable it in Minecraft: Options > Resource Packs")


if __name__ == "__main__":
    main()