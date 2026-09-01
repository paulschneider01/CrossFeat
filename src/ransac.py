"""Rigid RANSAC used by CrossFeat inlier-ridge adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RANSACResult:
    transform: np.ndarray
    inlier_mask: np.ndarray
    num_inliers: int
    inlier_ratio: float
    residual_errors: np.ndarray
    iterations: int

    @property
    def R(self) -> np.ndarray:
        return self.transform[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.transform[:3, 3]

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (points @ self.R.T) + self.t


class RANSACRigid:
    """RANSAC rigid transform estimator for Nx3 correspondences."""

    def __init__(
        self,
        threshold: float = 5.0,
        max_iterations: int = 1000,
        confidence: float = 0.99,
        min_inliers: int = 10,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.confidence = confidence
        self.min_inliers = min_inliers
        self.min_samples = 3

        if seed is not None and rng is not None:
            raise ValueError("Provide either seed or rng, not both.")
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(int(seed))
        else:
            self.rng = np.random.default_rng(int(np.random.randint(0, 2**32 - 1)))

    def fit(self, coords_a: np.ndarray, coords_b: np.ndarray) -> Optional[RANSACResult]:
        n_points = len(coords_a)
        if n_points < self.min_samples:
            return None

        best_num_inliers = 0
        best_inliers = None

        iterations = self.max_iterations
        iteration = 0
        while iteration < iterations:
            sample_idx = self.rng.choice(n_points, self.min_samples, replace=False)
            sample_a = coords_a[sample_idx]
            sample_b = coords_b[sample_idx]

            try:
                transform = self._estimate_rigid(sample_a, sample_b)
            except Exception:
                iteration += 1
                continue

            transformed = self._apply_transform(coords_a, transform)
            errors = np.linalg.norm(transformed - coords_b, axis=1)
            inliers = errors < self.threshold
            num_inliers = int(np.sum(inliers))

            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers

                if num_inliers >= self.min_inliers:
                    inlier_ratio = num_inliers / n_points
                    if inlier_ratio > 0:
                        p_no_outliers = inlier_ratio ** self.min_samples
                        if p_no_outliers >= 1.0:
                            iterations = iteration + 1
                        else:
                            p_no_outliers = float(np.clip(p_no_outliers, 1e-12, 1.0 - 1e-12))
                            denom = np.log(1.0 - p_no_outliers)
                            if denom != 0.0:
                                new_iterations = int(np.log(1.0 - self.confidence) / denom)
                                iterations = min(self.max_iterations, max(iteration + 1, new_iterations))

            iteration += 1

        if best_num_inliers < self.min_inliers or best_inliers is None:
            return None

        inlier_a = coords_a[best_inliers]
        inlier_b = coords_b[best_inliers]

        try:
            refined_transform = self._estimate_rigid(inlier_a, inlier_b)
        except Exception:
            return None

        transformed = self._apply_transform(coords_a, refined_transform)
        errors = np.linalg.norm(transformed - coords_b, axis=1)
        final_inliers = errors < self.threshold

        return RANSACResult(
            transform=refined_transform,
            inlier_mask=final_inliers,
            num_inliers=int(np.sum(final_inliers)),
            inlier_ratio=float(np.mean(final_inliers)),
            residual_errors=errors,
            iterations=iteration,
        )

    def _estimate_rigid(self, coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
        coords_a = np.asarray(coords_a, dtype=np.float64)
        coords_b = np.asarray(coords_b, dtype=np.float64)
        if coords_a.shape != coords_b.shape:
            raise ValueError(f"Shape mismatch: coords_a {coords_a.shape} != coords_b {coords_b.shape}")
        if coords_a.ndim != 2 or coords_a.shape[1] != 3:
            raise ValueError(f"Expected shape (N,3), got {coords_a.shape}")
        if coords_a.shape[0] < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} points, got {coords_a.shape[0]}")

        centroid_a = coords_a.mean(axis=0)
        centroid_b = coords_b.mean(axis=0)
        centered_a = coords_a - centroid_a
        centered_b = coords_b - centroid_b

        covariance = centered_a.T @ centered_b
        U, _, Vt = np.linalg.svd(covariance)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_b - R @ centroid_a

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = R
        transform[:3, 3] = t
        return transform

    @staticmethod
    def _apply_transform(coords: np.ndarray, transform: np.ndarray) -> np.ndarray:
        R = transform[:3, :3]
        t = transform[:3, 3]
        return (coords @ R.T) + t


# Backward-compatible alias used by existing import sites.
RANSAC3D = RANSACRigid
