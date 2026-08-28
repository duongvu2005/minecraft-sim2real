"""Datasets and transforms for the Minecraft frames and the Cityscapes validation set."""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from pipeline.blocks import IGNORE
from .config import MEAN, STD, EvalConfig, TrainConfig
from .remap import cityscapes_lut


def train_transform(cfg: TrainConfig) -> A.Compose:
    """Scale, crop, flip, then jitter color hard as implicit domain randomization."""
    return A.Compose([
        A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
        A.PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size,
                      border_mode=0, value=0, mask_value=IGNORE),
        A.RandomCrop(height=cfg.img_size, width=cfg.img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def val_transform(cfg: TrainConfig) -> A.Compose:
    """Letterbox to a square, no augmentation."""
    return A.Compose([
        A.LongestMaxSize(max_size=cfg.img_size),
        A.PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size,
                      border_mode=0, value=0, mask_value=IGNORE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def cityscapes_transform(cfg: EvalConfig) -> A.Compose:
    return A.Compose([
        A.Resize(height=cfg.height, width=cfg.width),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def split_pairs(
    rgb_dir: Path, mask_dir: Path, cfg: TrainConfig
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Match rgb_NNNN.png to mask_NNNN.png, then hold out cfg.n_val of them."""
    pairs = []
    for rgb_name in sorted(p.name for p in rgb_dir.glob("*.png")):
        mask_name = rgb_name.replace("rgb_", "mask_")
        if (mask_dir / mask_name).exists():
            pairs.append((rgb_name, mask_name))

    random.Random(cfg.seed).shuffle(pairs)
    return pairs[cfg.n_val:], pairs[:cfg.n_val]


class MinecraftSegDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], rgb_dir: Path,
                 mask_dir: Path, transform: A.Compose) -> None:
        self.pairs = pairs
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_name, mask_name = self.pairs[idx]
        image = np.array(Image.open(self.rgb_dir / rgb_name).convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / mask_name))
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].long()


def cityscapes_images(image_root: Path) -> list[Path]:
    return sorted(image_root.rglob("*_leftImg8bit.png"))


def cityscapes_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    relative = image_path.relative_to(image_root)
    return label_root / relative.with_name(
        relative.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
    )


class CityscapesValDataset(Dataset):
    """Cityscapes val, remapped to the four-class scheme."""

    def __init__(self, image_files: list[Path], image_root: Path,
                 label_root: Path, transform: A.Compose) -> None:
        self.image_files = image_files
        self.image_root = image_root
        self.label_root = label_root
        self.transform = transform
        self.lut = cityscapes_lut()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.image_files[idx]
        label_path = cityscapes_label_path(image_path, self.image_root, self.label_root)
        image = np.array(Image.open(image_path).convert("RGB"))
        # Remap before resizing, so the nearest-neighbour resize acts on the four
        # classes rather than on raw ids that would interpolate into each other.
        label = self.lut[np.array(Image.open(label_path))]
        out = self.transform(image=image, mask=label)
        return out["image"], out["mask"].long(), str(image_path)
