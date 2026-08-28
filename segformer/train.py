"""Model construction and the fine-tuning loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation

from pipeline.blocks import CLASS_NAMES
from .config import MODEL_NAME, TrainConfig


def build_model(num_classes: int = len(CLASS_NAMES),
                model_name: str = MODEL_NAME) -> SegformerForSemanticSegmentation:
    """
    Load the ADE20K checkpoint and swap its classifier for a four-class one.

    ignore_mismatched_sizes replaces only decode_head.classifier, the 1x1 conv
    from 768 channels to the class count. The encoder and the decoder's fusion
    layers keep their pretrained weights.
    """
    return SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label={i: name for i, name in enumerate(CLASS_NAMES)},
        label2id={name: i for i, name in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    )


def build_optimizer(model: SegformerForSemanticSegmentation,
                    cfg: TrainConfig) -> torch.optim.Optimizer:
    """Ten times the learning rate on the head, which is the only new part."""
    backbone: list[torch.nn.Parameter] = []
    head: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (head if "decode_head" in name else backbone).append(param)

    return torch.optim.AdamW(
        [{"params": backbone, "lr": cfg.lr_backbone},
         {"params": head, "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer,
                    total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear decay to zero, the schedule SegFormer is trained with."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / total_steps)
    )


def train(model: SegformerForSemanticSegmentation,
          train_loader: DataLoader,
          val_loader: DataLoader,
          evaluate_fn: Callable[..., dict[str, Any]],
          cfg: TrainConfig,
          device: torch.device,
          checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Fine-tune, keeping the checkpoint with the best held-out Minecraft mIoU."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg.epochs * len(train_loader))

    history: list[dict[str, Any]] = []
    best_miou = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for image, mask in progress:
            image, mask = image.to(device), mask.to(device)
            loss = model(pixel_values=image, labels=mask).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            progress.set_postfix({"loss": f"{loss.item():.3f}"})

        metrics = evaluate_fn(model, val_loader, device)
        metrics |= {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(metrics)
        print(f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f} | "
              f"val_loss={metrics['loss']:.4f} | val_mIoU={metrics['miou']:.4f}")

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "miou": best_miou}, checkpoint_dir / "best.pt")
            print(f"  -> new best, saved (mIoU={best_miou:.4f})")

    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best Minecraft-val mIoU: {best_miou:.4f}")
    return history


def load_checkpoint(path: Path, device: torch.device,
                    model_name: str = MODEL_NAME) -> SegformerForSemanticSegmentation:
    """Rebuild the architecture and load saved weights, so no training state carries over."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(model_name=model_name)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(f"Loaded epoch {checkpoint['epoch']}, Minecraft-val mIoU {checkpoint['miou']:.4f}")
    return model.to(device).eval()
