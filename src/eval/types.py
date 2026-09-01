from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvaluationResult:
    """Results for a single case evaluation."""

    case_id: str
    # Retrieval metrics
    top1_acc: float
    top5_acc: float
    top10_acc: float
    median_rank: float
    cosine_sim: float
    # Matching metrics
    num_samples: int
    num_matches: int
    num_inliers: int
    inlier_ratio: float  # = precision = num_inliers / num_matches
    recall: float  # = num_inliers / num_samples (fraction of GT pairs correctly matched)
    # Spatial coverage of inliers (0-1, higher = better distributed)
    spatial_coverage: float
    # TRE metrics (for aligned data or known transform)
    tre_mean: float
    tre_median: float
    tre_std: float
    # Timing
    time_s: float
    # Error (if any)
    error: str = ""
    # Slices used for evaluation (z-indices)
    slices_used: Optional[List[int]] = None
    # Success Rate @ thresholds (1 if mean_TRE <= threshold, else 0)
    # Per-case binary metric for comparison with MatchAnything/LightGlue papers
    sr_1px: float = 0.0
    sr_3px: float = 0.0
    sr_5px: float = 0.0
    sr_10px: float = 0.0
    # AUC @ thresholds (area under cumulative success-rate curve)
    # Per-case continuous metric that rewards fine accuracy
    auc_1px: float = 0.0
    auc_3px: float = 0.0
    auc_5px: float = 0.0
    auc_10px: float = 0.0
    # Raw per-match TRE values (for threshold sweeping in plots)
    tre_values: List[float] = field(default_factory=list)


def make_error_result(case_id: str, error: str, elapsed_time: float) -> EvaluationResult:
    """Create an error EvaluationResult with default values."""

    return EvaluationResult(
        case_id=case_id,
        top1_acc=0.0,
        top5_acc=0.0,
        top10_acc=0.0,
        median_rank=float("inf"),
        cosine_sim=0.0,
        num_samples=0,
        num_matches=0,
        num_inliers=0,
        inlier_ratio=0.0,
        recall=0.0,
        spatial_coverage=0.0,
        tre_mean=float("inf"),
        tre_median=float("inf"),
        tre_std=0.0,
        time_s=float(elapsed_time),
        error=str(error),
        sr_1px=0.0,
        sr_3px=0.0,
        sr_5px=0.0,
        sr_10px=0.0,
        auc_1px=0.0,
        auc_3px=0.0,
        auc_5px=0.0,
        auc_10px=0.0,
    )
