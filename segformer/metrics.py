"""Confusion matrix and the scores read off it."""

from __future__ import annotations

import torch


def fast_hist(pred: torch.Tensor, target: torch.Tensor, n: int) -> torch.Tensor:
    """
    Confusion matrix over one batch, counted in a single bincount.

    Entry (i, j) is the number of pixels whose ground truth is i and prediction
    is j. Pixels outside [0, n) on either side are dropped.
    """
    valid = (target >= 0) & (target < n) & (pred >= 0) & (pred < n)
    return torch.bincount(
        n * target[valid].long() + pred[valid].long(), minlength=n * n
    ).reshape(n, n)


def iou_from_hist(hist: torch.Tensor) -> torch.Tensor:
    intersection = torch.diag(hist).float()
    union = hist.sum(0).float() + hist.sum(1).float() - intersection
    return intersection / union.clamp(min=1)


def accuracy_from_hist(hist: torch.Tensor) -> torch.Tensor:
    """Per-class recall: of the pixels truly in class i, the fraction called i."""
    return torch.diag(hist).float() / hist.sum(1).float().clamp(min=1)
