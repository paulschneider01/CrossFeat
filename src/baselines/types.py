from __future__ import annotations

from typing import Any, Optional, Protocol

from src.eval.types import EvaluationResult
from src.viz import VisualizationData


class BaselineRunner(Protocol):
    name: str

    def config_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config summary for logging/output."""

    def evaluate_case(
        self,
        *,
        case: Any,
        modalities: list[str],
        extractor: Any,
        pca: Any,
        device: str,
        rotate_deg: float,
        translate_mm: float,
        affine_scale: float,
        affine_shear_deg: float,
        deterministic_transform: bool,
        n_samples_for_transform: int,
        inlier_threshold: float,
        seed: int,
        adapter: Any,
        return_viz_data: bool,
        two_d_only: bool,
        pair_sampling: str,
        max_kpts_per_slice: int,
        z_slices: Optional[list[int]] = None,
        ratio_thresh: float = 0.8,
    ) -> EvaluationResult | tuple[EvaluationResult, Optional[VisualizationData]]:
        """Evaluate baseline on one case."""
