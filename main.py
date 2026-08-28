"""Command line entry points for the Minecraft segmentation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline.blocks import CLASS_NAMES, IGNORE
from pipeline.process import build_dataset, class_distribution, print_class_distribution
from pipeline.resource_pack import build_pack, write_classification_log


def cmd_pack(args: argparse.Namespace) -> None:
    counts, assignments = build_pack(args.jar, args.output)
    log_path = args.output.parent / "classification_log.txt"
    write_classification_log(log_path, assignments)

    label = {**{i: n for i, n in enumerate(CLASS_NAMES)}, IGNORE: "ignore"}
    print(f"Resource pack written to {args.output.absolute()}")
    for cls in [0, 1, 2, IGNORE]:
        print(f"  {label[cls]:<12}{counts[cls]:>6} blocks")
    print(f"Classification log: {log_path.absolute()}")


def cmd_process(args: argparse.Namespace) -> None:
    n = build_dataset(args.screenshots, args.output, move=args.move)
    counts, total = class_distribution(args.output / "mask_label")
    print_class_distribution(counts, total)
    print(f"\n{n} pairs written to {args.output.absolute()}")


def cmd_stats(args: argparse.Namespace) -> None:
    mask_dir = args.dataset if args.dataset.name == "mask_label" else args.dataset / "mask_label"
    counts, total = class_distribution(mask_dir)
    print_class_distribution(counts, total)


def cmd_train(args: argparse.Namespace) -> None:
    # Imported here, not at module scope, so pack/process/stats/figures stay
    # runnable without torch installed.
    import torch
    from torch.utils.data import DataLoader

    from segformer.config import TrainConfig, device, seed_all
    from segformer.data import (MinecraftSegDataset, split_pairs, train_transform,
                                val_transform)
    from segformer.evaluate import evaluate
    from segformer.train import build_model, train

    cfg = TrainConfig()
    seed_all(cfg.seed)
    dev = device()
    print(f"device: {dev}")

    rgb_dir, mask_dir = args.data / "rgb", args.data / "mask_label"
    train_pairs, val_pairs = split_pairs(rgb_dir, mask_dir, cfg)
    print(f"train {len(train_pairs)} | val {len(val_pairs)}")

    train_ds = MinecraftSegDataset(train_pairs, rgb_dir, mask_dir, train_transform(cfg))
    val_ds = MinecraftSegDataset(val_pairs, rgb_dir, mask_dir, val_transform(cfg))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=2, pin_memory=True)

    model = build_model().to(dev)
    train(model, train_loader, val_loader, evaluate, cfg, dev, args.checkpoints)


def _cityscapes_loader(root: Path, cfg: "Any") -> tuple["Any", "Any"]:
    from torch.utils.data import DataLoader

    from segformer.data import (CityscapesValDataset, cityscapes_images,
                                cityscapes_transform)

    image_root, label_root = root / "leftImg8bit" / "val", root / "gtFine" / "val"
    images = cityscapes_images(image_root)
    print(f"{len(images)} Cityscapes val images")
    dataset = CityscapesValDataset(images, image_root, label_root,
                                   cityscapes_transform(cfg))
    return dataset, DataLoader(dataset, batch_size=cfg.batch_size, num_workers=2,
                               pin_memory=True)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from segformer.config import EvalConfig, device
    from segformer.evaluate import evaluate_cityscapes, print_results
    from segformer.train import load_checkpoint

    cfg, dev = EvalConfig(), device()
    _, loader = _cityscapes_loader(args.cityscapes, cfg)
    model = load_checkpoint(args.checkpoint, dev)
    results = evaluate_cityscapes(model, loader, dev)
    print_results("Minecraft -> Cityscapes", results)
    _save(results, args.out)


def cmd_baseline(args: argparse.Namespace) -> None:
    from transformers import SegformerForSemanticSegmentation

    from segformer.config import MODEL_NAME, EvalConfig, device
    from segformer.evaluate import evaluate_baseline, print_results
    from segformer.remap import ade20k_lut

    cfg, dev = EvalConfig(), device()
    _, loader = _cityscapes_loader(args.cityscapes, cfg)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(dev).eval()
    results = evaluate_baseline(model, loader, ade20k_lut(), dev)
    print_results("ADE20K -> Cityscapes (baseline)", results)
    _save(results, args.out)


def _save(results: dict, path: Path) -> None:
    import json

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in results.items()}
    path.write_text(json.dumps(serializable, indent=2))
    print(f"wrote {path}")


def cmd_figures(args: argparse.Namespace) -> None:
    """Runs from a clean clone: reads results/, needs no data and no GPU."""
    import json

    from segformer.plots import plot_training_curves

    history_path = args.results / "history.json"
    if history_path.exists():
        plot_training_curves(json.loads(history_path.read_text()),
                             args.figures / "training_curves.png")
    else:
        print(f"no {history_path}, skipping training curves")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pack", help="build the flat-color resource pack")
    p.add_argument("jar", type=Path, help="path to a Minecraft client jar")
    p.add_argument("-o", "--output", type=Path, default=Path("segmentation_pack"))
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("process", help="pair screenshots into a dataset")
    p.add_argument("screenshots", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--move", action="store_true", help="move instead of copying")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("stats", help="class distribution of a processed dataset")
    p.add_argument("dataset", type=Path)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("train", help="fine-tune SegFormer on the Minecraft frames")
    p.add_argument("data", type=Path, help="dataset directory holding rgb/ and mask_label/")
    p.add_argument("-c", "--checkpoints", type=Path, default=Path("checkpoints"))
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score a checkpoint on Cityscapes val")
    p.add_argument("cityscapes", type=Path, help="root holding leftImg8bit/ and gtFine/")
    p.add_argument("-c", "--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("-o", "--out", type=Path, default=Path("results/cityscapes.json"))
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("baseline", help="score the ADE20K model on Cityscapes val")
    p.add_argument("cityscapes", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("results/baseline.json"))
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("figures", help="rebuild figures from results/")
    p.add_argument("-r", "--results", type=Path, default=Path("results"))
    p.add_argument("-f", "--figures", type=Path, default=Path("figures"))
    p.set_defaults(func=cmd_figures)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
