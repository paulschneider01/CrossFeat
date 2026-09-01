"""Bridge SampleRecord -> CaseData for unified pipeline.

Provides two loading modes:
1. ``load_sample_as_case``: loads a single (1, H, W) slice/image from any format.
2. ``load_cases_from_manifest``: loads training-ready CaseData objects from a manifest.
   For NIfTI datasets (ReMIND), groups records by case and loads full 3D volumes.
   For 2D datasets, loads each record as individual (1, H, W) CaseData.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from src.io.case_loader import CaseData
from src.io.dataset_registry import DatasetManifest, SampleRecord
from src.io.nifti_orient import reorient_to_axial_first

logger = logging.getLogger(__name__)


# Thread-safe NIfTI proxy cache.  ``functools.lru_cache`` is NOT thread-safe
# in Python < 3.12 (concurrent dict/linked-list mutations), and this function
# is called from ``_load_2d_cases_parallel`` via ThreadPoolExecutor.
_nib_cache: dict[str, object] = {}
_nib_cache_lock = threading.Lock()
_NIB_CACHE_MAXSIZE = 64


def _cached_nib_load(path: str) -> object:
    """Thread-safe cache for nibabel lazy proxy objects.

    Uses double-checked locking to avoid contention on the fast path while
    remaining safe under concurrent access from worker threads.

    Returns a nibabel image object (lazy proxy, does not load full data).
    """
    existing = _nib_cache.get(path)
    if existing is not None:
        return existing

    import nibabel as nib

    with _nib_cache_lock:
        existing = _nib_cache.get(path)
        if existing is not None:
            return existing
        img = nib.load(path)
        # FIFO eviction: remove oldest entry when cache exceeds max size
        if len(_nib_cache) >= _NIB_CACHE_MAXSIZE:
            _nib_cache.pop(next(iter(_nib_cache)))
        _nib_cache[path] = img
        return img


def _load_nifti_slice(path: Path, z: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a single axial z-slice from a NIfTI file.

    Detects the axial (I/S) axis from the NIfTI affine so that datasets
    with different storage orientations (e.g. LPS for BraTS, RAS for
    RESECT vs IPL for ReMIND) all yield axial slices.

    Returns:
        (volume, affine) where volume has shape (1, H, W) float32.
    """
    from src.io.nifti_orient import get_axial_axis

    img = _cached_nib_load(str(path))
    affine = np.asarray(img.affine, dtype=np.float64)
    axial = get_axial_axis(affine)
    # Use dataobj proxy for lazy reads (avoids loading full volume)
    if axial == 0:
        slice_data = np.asarray(img.dataobj[z, :, :], dtype=np.float32)
    elif axial == 1:
        slice_data = np.asarray(img.dataobj[:, z, :], dtype=np.float32)
    else:
        slice_data = np.asarray(img.dataobj[:, :, z], dtype=np.float32)
    volume = slice_data[np.newaxis, :, :]  # (1, H, W)
    return volume, affine


def _load_jpg(path: Path, max_dim: int | None = 1024) -> np.ndarray:
    """Load a JPEG image as (1, H, W) float32.

    Args:
        path: Path to JPEG file.
        max_dim: If set, downscale so the longer edge <= max_dim.
            Default 1024 balances descriptor quality vs speed for
            high-res images (MegaDepth originals are 3000-6000px).
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load JPEG: {path}")
    if max_dim is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)[np.newaxis, :, :]


def _load_npy(path: Path, normalize_mode: str | None = None) -> np.ndarray:
    """Load a .npy array as (1, H, W) float32.

    Supported layouts: (H, W), (1, H, W), (H, W, C) where C in {1, 3, 4},
    and (C, H, W) where C in {1, 3, 4} and H, W > 4.

    ``normalize_mode`` is accepted for manifest compatibility but ignored.
    """
    _VALID_CHANNELS = (1, 3, 4)
    _BT601_WEIGHTS = (0.299, 0.587, 0.114)  # ITU-R BT.601 luma

    arr = np.load(str(path)).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]  # (H, W) -> (1, H, W)
    elif arr.ndim == 3 and arr.shape[2] in _VALID_CHANNELS:
        # (H, W, C) layout
        if arr.shape[2] == 1:
            arr = arr[:, :, 0][np.newaxis, :, :]
        else:
            arr = (
                _BT601_WEIGHTS[0] * arr[:, :, 0]
                + _BT601_WEIGHTS[1] * arr[:, :, 1]
                + _BT601_WEIGHTS[2] * arr[:, :, 2]
            )
            arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3 and arr.shape[0] in _VALID_CHANNELS and arr.shape[1] > 4 and arr.shape[2] > 4:
        # (C, H, W) layout
        if arr.shape[0] == 1:
            pass  # Already (1, H, W)
        else:
            arr = (
                _BT601_WEIGHTS[0] * arr[0]
                + _BT601_WEIGHTS[1] * arr[1]
                + _BT601_WEIGHTS[2] * arr[2]
            )
            arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        pass  # Already (1, H, W)
    else:
        raise ValueError(f"Unexpected .npy shape: {arr.shape} from {path}")

    return arr


def load_sample_as_case(record: SampleRecord) -> CaseData:
    """Load a SampleRecord into a CaseData with (1, H, W) volumes.

    Dispatch per file extension:
    - .nii/.nii.gz: cached nibabel lazy read of single z-slice
    - .png: cv2.imread grayscale
    - .npy: np.load with optional minmax normalization
    - .jpg/.jpeg: cv2.imread grayscale with optional max_dim downscaling

    Args:
        record: SampleRecord with modality_paths and optional slice_idx.

    Returns:
        CaseData with volumes dict (each modality -> (1, H, W) float32).

    Raises:
        FileNotFoundError: If a modality file doesn't exist.
    """
    volumes: dict[str, np.ndarray] = {}
    affines: dict[str, np.ndarray] = {}

    for mod_name, path in record.modality_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"File not found for sample={record.sample_id}, "
                f"modality={mod_name}: {path}"
            )

        suffix = "".join(path.suffixes).lower()

        if suffix in (".nii", ".nii.gz"):
            if record.slice_idx is None:
                raise ValueError(
                    f"slice_idx required for NIfTI files: "
                    f"sample={record.sample_id}, modality={mod_name}"
                )
            vol, aff = _load_nifti_slice(path, record.slice_idx)
            volumes[mod_name] = vol
            affines[mod_name] = aff
        elif suffix == ".png":
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f"Failed to load PNG: {path}")
            if raw.dtype == np.uint16:
                # Convert multi-channel uint16 to single-channel first
                if raw.ndim == 3:
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                img_f = raw.astype(np.float32)
                vmin, vmax = float(img_f.min()), float(img_f.max())
                if vmax - vmin > 1e-8:
                    img_f = (img_f - vmin) / (vmax - vmin) * 255.0
                else:
                    img_f = np.full_like(img_f, 128.0)
                volumes[mod_name] = img_f[np.newaxis, :, :]
            elif raw.ndim == 3:
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                volumes[mod_name] = gray.astype(np.float32)[np.newaxis, :, :]
            else:
                volumes[mod_name] = raw.astype(np.float32)[np.newaxis, :, :]
            affines[mod_name] = np.eye(4, dtype=np.float64)
        elif suffix == ".npy":
            volumes[mod_name] = _load_npy(path, record.normalize_mode)
            affines[mod_name] = np.eye(4, dtype=np.float64)
        elif suffix in (".jpg", ".jpeg"):
            max_dim = record.metadata.get("max_dim", 1024)
            volumes[mod_name] = _load_jpg(path, max_dim=max_dim)
            affines[mod_name] = np.eye(4, dtype=np.float64)
        elif suffix in (".tif", ".tiff"):
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f"Failed to load TIFF: {path}")
            if raw.ndim == 3:
                # Multi-channel (e.g. RGBA/RGB GeoTIFF) -> grayscale
                if raw.shape[2] == 4:
                    gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
                else:
                    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                volumes[mod_name] = gray.astype(np.float32)[np.newaxis, :, :]
            else:
                volumes[mod_name] = raw.astype(np.float32)[np.newaxis, :, :]
            affines[mod_name] = np.eye(4, dtype=np.float64)
        elif suffix == ".npz":
            data = np.load(str(path))
            arr = data[list(data.keys())[0]]
            if arr.ndim == 3:
                # Multi-channel -> grayscale via mean. For projected sensor
                # data (e.g. LiDAR [range, intensity, height]) the channels
                # are heterogeneous but averaging preserves local gradient
                # structure needed for descriptor extraction.
                gray = np.mean(arr[:, :, :min(arr.shape[2], 3)], axis=2)
                vol = gray.astype(np.float32)
            elif arr.ndim == 2:
                vol = arr.astype(np.float32)
            else:
                vol = arr.reshape(*arr.shape[-2:]).astype(np.float32)
            # Normalize to [0, 255] for descriptor extraction compatibility
            vmin, vmax = float(vol.min()), float(vol.max())
            if vmax - vmin > 1e-8:
                vol = (vol - vmin) / (vmax - vmin) * 255.0
            else:
                vol = np.full_like(vol, 128.0)
            volumes[mod_name] = vol[np.newaxis, :, :]
            affines[mod_name] = np.eye(4, dtype=np.float64)
        else:
            raise ValueError(
                f"Unsupported file format {suffix!r} for "
                f"sample={record.sample_id}, modality={mod_name}: {path}"
            )

    # Apply patch cropping if metadata specifies a sub-region.
    # Used by datasets with large images (e.g. WHU-OPT-SAR 3704x5556 -> 1024x1024).
    patch_size = record.metadata.get("patch_size")
    if patch_size is not None:
        y0 = record.metadata["patch_y0"]
        x0 = record.metadata["patch_x0"]
        for mod, vol in volumes.items():
            volumes[mod] = vol[:, y0 : y0 + patch_size, x0 : x0 + patch_size]

    # Resize modalities to a common (H, W) if they differ.
    # We resize to the *minimum* H and W to avoid hallucinating pixels.
    shapes = {mod: vol.shape[1:] for mod, vol in volumes.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        if record.metadata.get("require_same_shape", False):
            raise ValueError(
                f"Aligned modalities must have identical image dimensions for "
                f"sample={record.sample_id}: {shapes}"
            )
        min_h = min(s[0] for s in unique_shapes)
        min_w = min(s[1] for s in unique_shapes)
        for mod, vol in volumes.items():
            if vol.shape[1:] != (min_h, min_w):
                resized = cv2.resize(
                    vol[0], (min_w, min_h), interpolation=cv2.INTER_AREA,
                )
                volumes[mod] = resized[np.newaxis, :, :].astype(np.float32)

    return CaseData(
        case_id=record.sample_id,
        volumes=volumes,
        affines=affines,
        metadata=dict(record.metadata),
    )


def load_cases_from_manifest(
    manifest: DatasetManifest,
    split: str,
    max_cases: int | None = None,
) -> list[CaseData]:
    """Load CaseData objects from a manifest for training or evaluation.

    Loading strategy depends on the data format:
    - **NIfTI datasets** (records have ``slice_idx`` set): groups records by
      ``group_key`` (case_id) and loads full 3D volumes, since the extraction
      pipeline iterates over slices internally.
    - **2D datasets** (records have ``slice_idx=None``): loads each record as
      an individual (1, H, W) CaseData.

    Args:
        manifest: A DatasetManifest produced by ``build_manifest()``.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        max_cases: If set, load at most this many cases. For 2D datasets this
            truncates the record list *before* I/O, avoiding unnecessary image
            reads. Default ``None`` loads all cases.

    Returns:
        List of CaseData objects ready for ``extract_paired_descriptors()``.
    """
    if split == "train":
        records = manifest.train_records
    elif split == "val":
        records = manifest.val_records
    elif split == "test":
        records = manifest.test_records
    else:
        raise ValueError(f"Unknown split: {split!r}")

    if not records:
        return []

    # Check if this is a NIfTI dataset (records have slice_idx set).
    # Validate that ALL records are consistent to catch manifest bugs early.
    n_with_slice = sum(1 for r in records if r.slice_idx is not None)
    if 0 < n_with_slice < len(records):
        raise ValueError(
            f"Mixed slice_idx in manifest split={split!r}: "
            f"{n_with_slice} with slice_idx, "
            f"{len(records) - n_with_slice} without. "
            f"All records in a split must be consistently NIfTI or 2D."
        )
    has_slices = n_with_slice > 0

    if has_slices:
        return _load_nifti_cases(manifest, split)
    else:
        if max_cases is not None and len(records) > max_cases:
            logger.info(
                "Truncating %s records before loading: %d -> %d (max_cases=%d)",
                split, len(records), max_cases, max_cases,
            )
            records = records[:max_cases]
        return _load_2d_cases_parallel(records)


def _load_2d_cases_parallel(
    records: list[SampleRecord],
    max_workers: int = 8,
) -> list[CaseData]:
    """Load 2D records in parallel using a thread pool.

    ``cv2.imread`` releases the GIL, so threads achieve real I/O
    parallelism for PNG/NPY reads.  For small record lists the overhead
    of a pool isn't worth it, so we fall back to sequential loading.
    """
    if len(records) <= max_workers:
        return [load_sample_as_case(r) for r in records]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        cases = list(pool.map(load_sample_as_case, records))
    return cases


def _load_nifti_cases(
    manifest: DatasetManifest,
    split: str,
) -> list[CaseData]:
    """Load full 3D volumes for NIfTI datasets, grouped by case.

    All records sharing a ``group_key`` are assumed to point to the same NIfTI
    files (different slices of the same volume). We load the full volume once
    per case to enable multi-slice extraction.
    """
    import nibabel as nib

    groups = manifest.group_records_by_key(split)
    cases: list[CaseData] = []

    for group_key, group_records in sorted(groups.items()):
        # All records in a group share the same modality_paths
        first = group_records[0]
        volumes: dict[str, np.ndarray] = {}
        affines: dict[str, np.ndarray] = {}

        skip = False
        for mod_name, path in first.modality_paths.items():
            if not path.exists():
                logger.warning(
                    "Skipping case %s: file not found for modality %s: %s",
                    group_key, mod_name, path,
                )
                skip = True
                break
            img = nib.load(str(path))
            vol = img.get_fdata().astype(np.float32)
            affine = np.asarray(img.affine, dtype=np.float64)
            vol, affine = reorient_to_axial_first(vol, affine)
            volumes[mod_name] = vol
            affines[mod_name] = affine

        if skip:
            continue

        cases.append(
            CaseData(
                case_id=group_key,
                volumes=volumes,
                affines=affines,
            )
        )

    return cases
