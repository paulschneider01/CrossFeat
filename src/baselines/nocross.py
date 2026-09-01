"""No-crossing baseline: raw descriptor matching without a crossing model.

Extracts descriptors from both modalities using the same pipeline as CrossFeat
(same extractor, same PCA, same keypoint sampling), then matches via nearest
neighbor on L2-normalised descriptors. This isolates the contribution of the
crossing model by showing what pure descriptor similarity achieves.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from src.eval.types import EvaluationResult
from src.io.case_loader import CaseData
from src.viz import VisualizationData


class NoCrossBaselineRunner:
    """Baseline that matches raw (PCA-projected) descriptors without crossing."""

    name = "nocross"

    def config_dict(self) -> dict[str, Any]:
        return {"name": self.name}

    def evaluate_case(
        self,
        *,
        case: CaseData,
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
    ) -> Union[EvaluationResult, tuple[EvaluationResult, Optional[VisualizationData]]]:
        """Evaluate raw descriptor matching (no crossing model).

        Re-uses the same extraction / PCA / sampling pipeline as CrossFeat so
        the comparison is apples-to-apples.
        """
        # Import here to avoid circular dependency.
        from evaluate_core import evaluate_case_no_crossing

        result, viz_data = evaluate_case_no_crossing(
            case=case,
            modalities=modalities,
            extractor=extractor,
            pca=pca,
            device=device,
            rotate_deg=rotate_deg,
            translate_mm=translate_mm,
            affine_scale=affine_scale,
            affine_shear_deg=affine_shear_deg,
            deterministic_transform=deterministic_transform,
            n_samples=n_samples_for_transform,
            inlier_threshold=inlier_threshold,
            seed=seed,
            two_d_only=two_d_only,
            pair_sampling=pair_sampling,
            max_kpts_per_slice=max_kpts_per_slice,
            z_slices=z_slices,
            ratio_thresh=ratio_thresh,
        )

        if return_viz_data:
            return result, viz_data
        return result
