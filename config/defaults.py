"""Minimal dataset-root defaults used by training/evaluation scripts."""

from __future__ import annotations

# Dataset roots are intentionally explicit in public configs.
DATASET_ROOTS: dict[str, str] = {
    "paired_folders": "",
    "resect": "",
    "deliver": "",
    "qxs_saropt": "",
    "remind": "",
    "brats": "",
    "whu_opt_sar": "",
    "eventscape": "",
}
