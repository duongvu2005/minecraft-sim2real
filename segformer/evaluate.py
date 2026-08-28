"""Evaluation on held-out Minecraft frames, on Cityscapes, and for the ADE20K baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pipeline.blocks import CLASS_NAMES, IGNORE
from .metrics import accuracy_from_hist, fast_hist, iou_from_hist

N_CLASSES = len(CLASS_NAMES)


def _predict(model: Any, image: torch.Tensor, size: torch.Size) -> torch.Tensor:
    """SegFormer emits logits at a quarter resolution, so upsample before arg-maxing."""
    logits = model(pixel_values=image).logits
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False).argmax(1)


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Loss and IoU on the held-out Minecraft split."""
    model.eval()
    hist = torch.zeros(N_CLASSES, N_CLASSES, device=device)
    # Weighted by images, not batches, so a ragged final batch does not count as
    # much as a full one.
    loss_sum = 0.0
    n_images = 0

    for image, mask in loader:
        image, mask = image.to(device), mask.to(device)
        out = model(pixel_values=image, labels=mask)
        loss_sum += out.loss.item() * image.shape[0]
        n_images += image.shape[0]
        logits = F.interpolate(out.logits, size=mask.shape[-2:],
                               mode="bilinear", align_corners=False)
        labelled = mask != IGNORE
        hist += fast_hist(logits.argmax(1)[labelled], mask[labelled], N_CLASSES)

    iou = iou_from_hist(hist)
    return {"loss": loss_sum / max(n_images, 1), "miou": iou.mean().item(),
            "per_class_iou": iou.cpu().tolist()}


@torch.no_grad()
def evaluate_cityscapes(model: Any, loader: DataLoader,
                        device: torch.device) -> dict[str, Any]:
    """Zero-shot transfer of the four-class model to Cityscapes."""
    model.eval()
    hist = torch.zeros(N_CLASSES, N_CLASSES, device=device, dtype=torch.float64)

    for image, mask, _ in tqdm(loader, desc="Cityscapes eval"):
        image, mask = image.to(device), mask.to(device)
        pred = _predict(model, image, mask.shape[-2:])
        labelled = mask != IGNORE
        hist += fast_hist(pred[labelled], mask[labelled], N_CLASSES).double()

    return _summarize(hist)


@torch.no_grad()
def evaluate_baseline(model_150: Any, loader: DataLoader, ade_lut: np.ndarray,
                      device: torch.device) -> dict[str, Any]:
    """
    The ADE20K model scored on Cityscapes through a fixed class mapping.

    The model predicts over 150 ADE classes, most of which have no counterpart in
    the four-class scheme. Those pixels are dropped rather than counted wrong, so
    the baseline is scored only where it made a mappable prediction; `coverage`
    is the fraction of labelled pixels that survived.
    """
    model_150.eval()
    hist = torch.zeros(N_CLASSES, N_CLASSES, device=device, dtype=torch.float64)
    lut = torch.from_numpy(ade_lut.astype(np.int64)).to(device)
    labelled_total = scored_total = 0

    for image, mask, _ in tqdm(loader, desc="Baseline eval"):
        image, mask = image.to(device), mask.to(device)
        pred = lut[_predict(model_150, image, mask.shape[-2:])]

        labelled = mask != IGNORE
        scored = labelled & (pred != IGNORE)
        labelled_total += int(labelled.sum())
        scored_total += int(scored.sum())
        hist += fast_hist(pred[scored], mask[scored], N_CLASSES).double()

    results = _summarize(hist)
    results["coverage"] = scored_total / max(labelled_total, 1)
    return results


def _summarize(hist: torch.Tensor) -> dict[str, Any]:
    iou = iou_from_hist(hist)
    return {
        "miou": iou.mean().item(),
        "per_class_iou": iou.cpu().tolist(),
        "per_class_acc": accuracy_from_hist(hist).cpu().tolist(),
        "hist": hist.cpu().numpy(),
    }


def print_results(title: str, results: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(f"mIoU: {results['miou']:.4f}")
    print(f"{'class':<12}{'IoU':>9}{'Acc':>9}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12}{results['per_class_iou'][i]:>9.4f}"
              f"{results['per_class_acc'][i]:>9.4f}")
    if "coverage" in results:
        print(f"coverage: {results['coverage']:.2%} of labelled pixels scored")
