"""Inlier-ridge test-time adapter used by CrossFeat."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge

from src.ransac import RANSAC3D


class BaseAdapter(ABC):
    @abstractmethod
    def adapt(
        self,
        d_crossed: np.ndarray,
        d_target: np.ndarray,
        coords_a: Optional[np.ndarray] = None,
        coords_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_config()})"


def mutual_nn_matches(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    distances = cdist(desc_a, desc_b, metric=metric)
    forward = np.argmin(distances, axis=1)
    backward = np.argmin(distances, axis=0)
    mutual = backward[forward] == np.arange(len(desc_a))
    idx_a = np.where(mutual)[0]
    idx_b = forward[idx_a]
    return idx_a, idx_b


class InlierRidgeAdapter(BaseAdapter):
    def __init__(
        self,
        ransac_threshold: float = 5.0,
        ransac_min_inliers: int = 20,
        ridge_alpha: float = 1.0,
        n_iterations: int = 1,
        min_inlier_ratio: float = 0.1,
        normalize_output: bool = True,
        residual_mode: bool = True,
        residual_alpha: float = 0.5,
        use_adain_warmup: bool = True,
    ):
        self.ransac_threshold = ransac_threshold
        self.ransac_min_inliers = ransac_min_inliers
        self.ridge_alpha = ridge_alpha
        self.n_iterations = n_iterations
        self.min_inlier_ratio = min_inlier_ratio
        self.normalize_output = normalize_output
        self.residual_mode = residual_mode
        self.residual_alpha = residual_alpha
        self.use_adain_warmup = use_adain_warmup
        self.ransac = RANSAC3D(
            threshold=ransac_threshold,
            min_inliers=ransac_min_inliers,
            max_iterations=1000,
        )
        self._last_debug_info = {}

    def _apply_adain(
        self,
        d_crossed: np.ndarray,
        d_target: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        mu_crossed = d_crossed.mean(axis=0, keepdims=True)
        std_crossed = d_crossed.std(axis=0, keepdims=True) + eps
        mu_target = d_target.mean(axis=0, keepdims=True)
        std_target = d_target.std(axis=0, keepdims=True) + eps
        d_norm = (d_crossed - mu_crossed) / std_crossed
        d_adapted = d_norm * std_target + mu_target
        norms = np.linalg.norm(d_adapted, axis=1, keepdims=True)
        return d_adapted / (norms + eps)

    def adapt(
        self,
        d_crossed: np.ndarray,
        d_target: np.ndarray,
        coords_a: Optional[np.ndarray] = None,
        coords_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._last_debug_info = {
            "iterations_completed": 0,
            "inliers_per_iteration": [],
            "inlier_ratio_per_iteration": [],
            "calibration_applied": False,
        }
        if coords_a is None or coords_b is None:
            return d_crossed.copy()

        d_current = d_crossed.copy()
        if self.use_adain_warmup:
            d_current = self._apply_adain(d_current, d_target)

        for iteration in range(self.n_iterations):
            idx_a, idx_b = mutual_nn_matches(d_current, d_target)
            if len(idx_a) < self.ransac_min_inliers:
                break

            result = self.ransac.fit(coords_a[idx_a], coords_b[idx_b])
            if result is None:
                break

            inlier_ratio = result.inlier_ratio
            self._last_debug_info["inliers_per_iteration"].append(result.num_inliers)
            self._last_debug_info["inlier_ratio_per_iteration"].append(inlier_ratio)
            if result.num_inliers < self.ransac_min_inliers:
                break
            if inlier_ratio < self.min_inlier_ratio:
                break

            inlier_idx_a = idx_a[result.inlier_mask]
            inlier_idx_b = idx_b[result.inlier_mask]
            d_inlier_crossed = d_current[inlier_idx_a]
            d_inlier_target = d_target[inlier_idx_b]

            try:
                if self.residual_mode:
                    delta_target = d_inlier_target - d_inlier_crossed
                    ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
                    ridge.fit(d_inlier_crossed, delta_target)
                    delta_pred = ridge.predict(d_current)
                    d_current = d_current + self.residual_alpha * delta_pred
                else:
                    ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
                    ridge.fit(d_inlier_crossed, d_inlier_target)
                    d_current = ridge.predict(d_current)
                self._last_debug_info["calibration_applied"] = True
            except Exception:
                break

            self._last_debug_info["iterations_completed"] = iteration + 1
            norms = np.linalg.norm(d_current, axis=1, keepdims=True)
            d_current = d_current / (norms + 1e-8)

        if self.normalize_output:
            norms = np.linalg.norm(d_current, axis=1, keepdims=True)
            d_current = d_current / (norms + 1e-8)
        return d_current.astype(np.float32)

    def get_debug_info(self) -> Dict[str, Any]:
        return self._last_debug_info.copy()

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "inlier_ridge",
            "ransac_threshold": self.ransac_threshold,
            "ransac_min_inliers": self.ransac_min_inliers,
            "ridge_alpha": self.ridge_alpha,
            "n_iterations": self.n_iterations,
            "min_inlier_ratio": self.min_inlier_ratio,
            "residual_mode": self.residual_mode,
            "residual_alpha": self.residual_alpha,
            "use_adain_warmup": self.use_adain_warmup,
        }
