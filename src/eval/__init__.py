"""Evaluation helpers used by CrossFeat."""

from .metrics import TREStats, compute_retrieval_accuracy, compute_sr_auc_from_tre, compute_tre_statistics
from .types import EvaluationResult, make_error_result

__all__ = [
    "TREStats",
    "compute_retrieval_accuracy",
    "compute_sr_auc_from_tre",
    "compute_tre_statistics",
    "EvaluationResult",
    "make_error_result",
]
