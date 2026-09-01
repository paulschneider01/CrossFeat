"""Metrics used by CrossFeat training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_retrieval_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    max_n: int = 2000,
) -> Dict[str, float]:
    """Compute top-k retrieval accuracy (percent) for paired descriptors."""
    n = min(len(pred), max_n)
    if n == 0:
        return {"top1": 0.0, "top5": 0.0, "top10": 0.0}

    pred = np.asarray(pred[:n], dtype=np.float32)
    target = np.asarray(target[:n], dtype=np.float32)

    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)

    dists = 2.0 - 2.0 * (pred @ target.T)
    np.maximum(dists, 0.0, out=dists)

    gt = np.arange(n, dtype=np.int64)
    top1_idx = np.argmin(dists, axis=1)
    top1_hits = top1_idx == gt

    k5 = min(5, n)
    if k5 == n:
        top5_hits = np.ones(n, dtype=bool)
    else:
        top5_idx = np.argpartition(dists, kth=k5 - 1, axis=1)[:, :k5]
        top5_hits = np.any(top5_idx == gt[:, None], axis=1)

    k10 = min(10, n)
    if k10 == n:
        top10_hits = np.ones(n, dtype=bool)
    else:
        top10_idx = np.argpartition(dists, kth=k10 - 1, axis=1)[:, :k10]
        top10_hits = np.any(top10_idx == gt[:, None], axis=1)

    return {
        "top1": float(np.mean(top1_hits) * 100.0),
        "top5": float(np.mean(top5_hits) * 100.0),
        "top10": float(np.mean(top10_hits) * 100.0),
    }


def _compute_auc(errors: np.ndarray, thresholds: List[float]) -> List[float]:
    if len(errors) == 0:
        return [0.0] * len(thresholds)

    errors_sorted = np.sort(errors)
    n = len(errors_sorted)
    recall = (np.arange(n) + 1) / n

    errors_sorted = np.r_[0.0, errors_sorted]
    recall = np.r_[0.0, recall]

    aucs: List[float] = []
    for threshold in thresholds:
        idx = np.searchsorted(errors_sorted, threshold, side="right")
        if idx <= 1:
            aucs.append(0.0)
            continue

        e = errors_sorted[:idx]
        r = recall[:idx]
        if e[-1] < threshold:
            e = np.r_[e, threshold]
            r = np.r_[r, r[-1]]

        dx = np.diff(e)
        y_avg = (r[:-1] + r[1:]) / 2.0
        aucs.append(float(np.sum(dx * y_avg)) / threshold)
    return aucs


@dataclass
class TREStats:
    mean: float
    median: float
    std: float
    sr_values: List[float]
    auc_values: List[float]
    thresholds: List[float]


def compute_tre_statistics(
    tre_values: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> TREStats:
    if thresholds is None:
        thresholds = [1.0, 3.0, 5.0, 10.0]

    tre_values = np.asarray(tre_values)
    finite_mask = np.isfinite(tre_values)
    tre_finite = tre_values[finite_mask]

    if tre_finite.size == 0:
        if tre_values.size > 0:
            sr_fill = [float("nan")] * len(thresholds)
            auc_fill = [float("nan")] * len(thresholds)
        else:
            sr_fill = [0.0] * len(thresholds)
            auc_fill = [0.0] * len(thresholds)
        return TREStats(
            mean=float("inf"),
            median=float("inf"),
            std=0.0,
            sr_values=sr_fill,
            auc_values=auc_fill,
            thresholds=thresholds,
        )

    tre_mean = float(np.mean(tre_finite))
    tre_median = float(np.median(tre_finite))
    tre_std = float(np.std(tre_finite))
    sr_values = [1.0 if tre_mean <= threshold else 0.0 for threshold in thresholds]
    auc_values = _compute_auc(tre_finite, thresholds)

    return TREStats(
        mean=tre_mean,
        median=tre_median,
        std=tre_std,
        sr_values=sr_values,
        auc_values=auc_values,
        thresholds=thresholds,
    )


def compute_sr_auc_from_tre(
    tre_values: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    stats = compute_tre_statistics(tre_values, thresholds)
    return stats.sr_values, stats.auc_values
