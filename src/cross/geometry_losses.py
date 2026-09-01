"""Geometry-preservation losses used by the VAE training pipeline."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseDistancePreservationLoss(nn.Module):
    def __init__(self, normalize_scale: bool = True):
        super().__init__()
        self.normalize_scale = normalize_scale

    def forward(self, d_before: torch.Tensor, d_after: torch.Tensor) -> torch.Tensor:
        d_before = F.normalize(d_before, p=2, dim=1)
        d_after = F.normalize(d_after, p=2, dim=1)

        dist_before = torch.cdist(d_before, d_before, p=2)
        dist_after = torch.cdist(d_after, d_after, p=2)

        if self.normalize_scale:
            dist_before = dist_before / (dist_before.mean() + 1e-8)
            dist_after = dist_after / (dist_after.mean() + 1e-8)

        return F.mse_loss(dist_before, dist_after)


class SimilarityMatrixPreservationLoss(nn.Module):
    def __init__(self, top_k: Optional[int] = None, exclude_diagonal: bool = True):
        super().__init__()
        self.top_k = top_k
        self.exclude_diagonal = exclude_diagonal

    def forward(self, d_before: torch.Tensor, d_after: torch.Tensor) -> torch.Tensor:
        batch_size = d_before.shape[0]

        d_before = F.normalize(d_before, p=2, dim=1)
        d_after = F.normalize(d_after, p=2, dim=1)

        sim_before = d_before @ d_before.T
        sim_after = d_after @ d_after.T

        if self.exclude_diagonal:
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=d_before.device)
            sim_before = sim_before[mask]
            sim_after = sim_after[mask]

        if self.top_k is not None and self.top_k < batch_size and not self.exclude_diagonal:
            sim_before_mat = d_before @ d_before.T
            sim_after_mat = d_after @ d_after.T
            _, top_idx = sim_before_mat.topk(self.top_k, dim=1)
            sim_before = torch.gather(sim_before_mat, 1, top_idx)
            sim_after = torch.gather(sim_after_mat, 1, top_idx)

        return F.mse_loss(sim_before, sim_after)


__all__ = ["PairwiseDistancePreservationLoss", "SimilarityMatrixPreservationLoss"]
