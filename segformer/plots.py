"""Every figure. Takes arrays, never a model, so it runs without torch or the data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from pipeline.blocks import CLASS_NAMES, IGNORE

# Sky is drawn light blue rather than the pack's cyan.
PALETTE = np.array(
    [[255, 0, 0], [0, 0, 255], [0, 255, 0], [135, 206, 235]], dtype=np.uint8
)
IGNORE_COLOR = (80, 80, 80)


def colorize(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for index, color in enumerate(PALETTE):
        out[labels == index] = color
    out[labels == IGNORE] = IGNORE_COLOR
    return out


def denormalize(image: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Normalized CHW array back to an HWC image."""
    array = image.transpose(1, 2, 0) * np.asarray(std) + np.asarray(mean)
    return array.clip(0, 1)


def plot_training_curves(history: list[dict[str, Any]], path: Path | None = None) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [h["miou"] for h in history])
    axes[1].set_title("Val mIoU")
    axes[1].set_xlabel("epoch")
    _finish(fig, path)


def plot_confusion_matrix(hist: np.ndarray, path: Path | None = None) -> None:
    """Rows are ground truth and sum to one, so the diagonal is per-class recall."""
    normalized = hist / hist.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Cityscapes confusion matrix (row-normalized)")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                    color="white" if normalized[i, j] > 0.5 else "black", fontsize=10)
    fig.colorbar(image, ax=ax)
    _finish(fig, path)


def plot_qualitative(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                     titles: Sequence[str] | None = None,
                     path: Path | None = None) -> None:
    """One row per sample: image, ground truth, prediction."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4 * len(rows)), squeeze=False)
    for row, (image, truth, prediction) in enumerate(rows):
        panels = [(image, titles[row] if titles else "RGB"),
                  (colorize(truth), "Ground truth"),
                  (colorize(prediction), "Prediction")]
        for col, (panel, title) in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    _finish(fig, path)


def _finish(fig: plt.Figure, path: Path | None) -> None:
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)
