from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.io.case_adapter import load_sample_as_case
from src.io.dataset_registry import build_manifest


def _write_image(path: Path, shape: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.arange(shape[0] * shape[1], dtype=np.uint16).reshape(shape) % 256
    assert cv2.imwrite(str(path), image.astype(np.uint8))


def _make_layout(root: Path) -> None:
    for split in ("train", "val", "test"):
        for modality in ("a", "b"):
            _write_image(root / split / modality / "nested" / "sample.png")


def test_paired_folders_builds_all_splits(tmp_path: Path) -> None:
    _make_layout(tmp_path)

    manifest = build_manifest("paired_folders", tmp_path, ["a", "b"])

    assert len(manifest.train_records) == 1
    assert len(manifest.val_records) == 1
    assert len(manifest.test_records) == 1
    assert manifest.train_records[0].sample_id == "train/nested/sample.png"
    assert set(manifest.train_records[0].modality_paths) == {"a", "b"}


def test_paired_folders_rejects_missing_pair(tmp_path: Path) -> None:
    _make_layout(tmp_path)
    _write_image(tmp_path / "val" / "b" / "nested" / "other.png")
    (tmp_path / "val" / "b" / "nested" / "sample.png").unlink()

    with pytest.raises(ValueError, match="Unpaired files"):
        build_manifest("paired_folders", tmp_path, ["a", "b"])


def test_paired_folders_rejects_shape_mismatch(tmp_path: Path) -> None:
    _make_layout(tmp_path)
    _write_image(tmp_path / "train" / "b" / "nested" / "sample.png", (48, 64))
    manifest = build_manifest("paired_folders", tmp_path, ["a", "b"])

    with pytest.raises(ValueError, match="identical image dimensions"):
        load_sample_as_case(manifest.train_records[0])


def test_paired_folders_does_not_hide_jpeg_shape_mismatch(tmp_path: Path) -> None:
    _make_layout(tmp_path)
    _write_image(tmp_path / "train" / "a" / "large.jpg", (80, 100))
    _write_image(tmp_path / "train" / "b" / "large.jpg", (160, 200))
    manifest = build_manifest("paired_folders", tmp_path, ["a", "b"])
    record = next(r for r in manifest.train_records if r.sample_id.endswith("large.jpg"))

    with pytest.raises(ValueError, match="identical image dimensions"):
        load_sample_as_case(record)
