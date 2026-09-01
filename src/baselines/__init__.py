"""Baseline methods used for evaluation comparisons."""

from src.baselines.defaults import add_baseline_args
from src.baselines.nocross import NoCrossBaselineRunner

__all__ = [
    "add_baseline_args",
    "NoCrossBaselineRunner",
]
