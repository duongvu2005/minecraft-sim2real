"""Training and evaluation settings."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

# ImageNet statistics, which the pretrained SegFormer weights expect.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    n_val: int = 30
    img_size: int = 512
    batch_size: int = 8
    epochs: int = 40
    lr_backbone: float = 6e-6
    lr_head: float = 6e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0


@dataclass(frozen=True)
class EvalConfig:
    """Cityscapes is evaluated at half its native size, keeping the 2:1 aspect."""

    width: int = 1024
    height: int = 512
    batch_size: int = 4


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
