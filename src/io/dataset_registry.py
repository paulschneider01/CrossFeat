"""Dataset registry and manifest types for multi-dataset support.

Provides a manifest-based (not discovery-based) registry with explicit
per-modality file paths. All datasets (including ReMIND) produce the same
SampleRecord/DatasetManifest types, enabling a unified pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Registry of dataset manifest builders
_DATASET_BUILDERS: dict[str, Callable[..., DatasetManifest]] = {}


@dataclass
class SampleRecord:
    """One sample (image or z-slice) across all modalities.

    Attributes:
        sample_id: Unique identifier. For ReMIND: "Case013:z042". For 2D: "285".
        modality_paths: modality_name -> absolute file path.
        split: "train" | "val" | "test".
        group_key: Grouping key for leakage-safe splitting (case_id, category, scene).
        slice_idx: z-slice index for NIfTI (None for 2D datasets).
        normalize_mode: "minmax" for .npy float arrays, None for PNG/NIfTI.
        metadata: Dataset-specific parameters forwarded to loaders (e.g. max_dim for JPEG/HDF5).
    """

    sample_id: str
    modality_paths: dict[str, Path]
    split: str
    group_key: str
    slice_idx: Optional[int] = None
    normalize_mode: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetManifest:
    """Complete manifest for a dataset.

    Attributes:
        name: Dataset name ("deliver", "eventscape", "remind", "resect",
            "brats", "whu_opt_sar", "qxs_saropt").
        root: Root directory of the dataset.
        modalities: Ordered list of modality names.
        is_2d: True for all datasets (ReMIND is slice-level too).
        image_diagonal: Image diagonal in pixels for normalized TRE (None if not applicable).
        records: List of sample records.
    """

    name: str
    root: Path
    modalities: list[str]
    is_2d: bool
    image_diagonal: Optional[float] = None
    records: list[SampleRecord] = field(default_factory=list)

    @property
    def train_records(self) -> list[SampleRecord]:
        return [r for r in self.records if r.split == "train"]

    @property
    def val_records(self) -> list[SampleRecord]:
        return [r for r in self.records if r.split == "val"]

    @property
    def test_records(self) -> list[SampleRecord]:
        return [r for r in self.records if r.split == "test"]

    def records_for_split(self, split: str) -> list[SampleRecord]:
        return [r for r in self.records if r.split == split]

    def group_records_by_key(self, split: str) -> dict[str, list[SampleRecord]]:
        """Group records by group_key within a split."""
        groups: dict[str, list[SampleRecord]] = {}
        for r in self.records_for_split(split):
            groups.setdefault(r.group_key, []).append(r)
        return groups


def register_dataset(name: str) -> Callable:
    """Decorator to register a dataset manifest builder."""

    def decorator(fn: Callable[..., DatasetManifest]) -> Callable[..., DatasetManifest]:
        _DATASET_BUILDERS[name] = fn
        return fn

    return decorator


def build_manifest(
    dataset_name: str,
    data_root: str | Path,
    modalities: list[str],
    **kwargs: object,
) -> DatasetManifest:
    """Build a manifest for the specified dataset.

    Args:
        dataset_name: Name of the registered dataset.
        data_root: Root directory of the dataset.
        modalities: List of modality names to include.
        **kwargs: Additional dataset-specific arguments (e.g., use_crop for ReMIND).

    Returns:
        DatasetManifest with all records populated.

    Raises:
        KeyError: If dataset_name is not registered.
    """
    if dataset_name not in _DATASET_BUILDERS:
        available = ", ".join(sorted(_DATASET_BUILDERS.keys()))
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. Available: {available}"
        )
    return _DATASET_BUILDERS[dataset_name](
        data_root=Path(data_root),
        modalities=modalities,
        **kwargs,
    )


# Import dataset loader definitions so @register_dataset decorators run.
import src.io.dataset_loaders as _dataset_loaders

_dataset_loaders
