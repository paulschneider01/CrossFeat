"""Descriptor extraction utilities for training."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Tuple

import numpy as np
from scipy.ndimage import sobel
from tqdm import tqdm

from src.io.case_loader import CaseData
from src.utils.transforms import (
    apply_transform_to_coords,
    generate_random_affine_in_plane,
    generate_random_transform,
    generate_random_transform_in_plane,
    transform_keypoints_with_z,
    warp_volume_rigid,
    warp_volume_affine,
)
from src.utils.sampling import (
    sample_coords_sift_union_from_roi,
    sample_keypoints_2d_dual,
    sample_keypoints_2d_ref,
)
from src.utils.roi_preprocessing import compute_safe_crop_bounds, safe_preprocess_roi

_PAIR_SAMPLING_MODES = {"random", "random_with_kp", "sift_union", "2d_sift_ref", "2d_sift_dual"}

# Default keypoint size when no nearby detected keypoint is available
_DEFAULT_KP_SIZE = 16.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parallelism helpers
# ---------------------------------------------------------------------------

@contextmanager
def _opencv_thread_limit(n: int) -> Generator[None, None, None]:
    """Temporarily set OpenCV thread count, restoring on exit."""
    try:
        import cv2
        prev = cv2.getNumThreads()
        cv2.setNumThreads(n)
    except ImportError:
        yield
        return
    try:
        yield
    finally:
        try:
            cv2.setNumThreads(prev)
        except Exception:
            logger.debug("Failed to restore cv2 thread count", exc_info=True)


def _prepare_parallel_extractor(extractor: object, effective_workers: int, log: logging.Logger) -> int:
    """Validate extractor for parallel use. Returns effective_workers (may reduce to 1)."""
    if effective_workers <= 1:
        return 1

    # GPU extractors don't benefit from threading: Python's GIL serialises
    # CPU work and duplicating models on the same GPU adds contention.
    ext_device = str(getattr(extractor, "device", "cpu")).lower()
    if "cuda" in ext_device:
        log.info(
            "Extractor %s runs on GPU (%s); multi-threaded GPU inference is "
            "slower — using sequential extraction (requested %d workers)",
            type(extractor).__name__,
            ext_device,
            effective_workers,
        )
        return 1

    clone_fn = getattr(extractor, "clone", None)
    if not callable(clone_fn):
        log.warning(
            "Extractor %s has no callable clone(); falling back to sequential "
            "(requested %d workers)",
            type(extractor).__name__,
            effective_workers,
        )
        return 1
    try:
        test_clone = clone_fn()
    except Exception:
        log.warning(
            "Extractor %s clone() raised; falling back to sequential",
            type(extractor).__name__,
            exc_info=True,
        )
        return 1
    if test_clone is extractor:
        log.warning("Extractor clone() returned self; falling back to sequential")
        return 1
    return effective_workers


@dataclass
class _CaseResult:
    """Per-case extraction result for parallel aggregation."""

    descs_a: list[np.ndarray] = field(default_factory=list)
    descs_b: list[np.ndarray] = field(default_factory=list)
    skipped_low_mask: int = 0
    skipped_low_transform: int = 0
    skipped_low_valid: int = 0
    # Local properties (populated only when return_local_props=True)
    local_gradients_a: list[np.ndarray] = field(default_factory=list)
    local_gradients_b: list[np.ndarray] = field(default_factory=list)
    local_intensities_a: list[np.ndarray] = field(default_factory=list)
    local_intensities_b: list[np.ndarray] = field(default_factory=list)
    local_coords_a: list[np.ndarray] = field(default_factory=list)
    local_coords_b: list[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared extraction helpers
# ---------------------------------------------------------------------------

def _coords_from_keypoints_with_z(kps_with_z: list[tuple[int, object]], shape: np.ndarray) -> np.ndarray:
    """Convert (z, cv2.KeyPoint) tuples into integer voxel coordinates [z, y, x]."""
    coords = np.zeros((len(kps_with_z), 3), dtype=np.int32)
    z_max = int(shape[0] - 1)
    y_max = int(shape[1] - 1)
    x_max = int(shape[2] - 1)
    for i, (z, kp) in enumerate(kps_with_z):
        z_i = int(np.clip(int(round(float(z))), 0, z_max))
        x_f, y_f = kp.pt
        y_i = int(np.clip(int(round(float(y_f))), 0, y_max))
        x_i = int(np.clip(int(round(float(x_f))), 0, x_max))
        coords[i] = (z_i, y_i, x_i)
    return coords


def _extract_local_intensity_and_gradient(
    vol: np.ndarray,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute probe-aligned local intensity and gradient magnitude at coordinates."""
    n = int(len(coords))
    if n == 0:
        zeros = np.zeros((0,), dtype=np.float32)
        return zeros, zeros.copy()

    shape = np.array(vol.shape)
    intensities = np.zeros(n, dtype=np.float32)
    gradients = np.zeros(n, dtype=np.float32)
    grad_cache: dict[int, np.ndarray] = {}

    for i in range(n):
        z_i = int(coords[i, 0])
        y_i = int(coords[i, 1])
        x_i = int(coords[i, 2])
        z_i = int(np.clip(z_i, 0, shape[0] - 1))
        y_i = int(np.clip(y_i, 0, shape[1] - 1))
        x_i = int(np.clip(x_i, 0, shape[2] - 1))

        # Probe-B-aligned local intensity: 5x5x5 patch mean.
        z_lo = max(0, z_i - 2)
        z_hi = min(shape[0], z_i + 3)
        y_lo = max(0, y_i - 2)
        y_hi = min(shape[1], y_i + 3)
        x_lo = max(0, x_i - 2)
        x_hi = min(shape[2], x_i + 3)
        patch = vol[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
        intensities[i] = float(patch.mean())

        # Probe-C-aligned local gradient: Sobel magnitude on axial slice + 5x5 patch mean.
        if z_i not in grad_cache:
            slc = vol[z_i].astype(np.float64, copy=False)
            sx = sobel(slc, axis=1)
            sy = sobel(slc, axis=0)
            grad_cache[z_i] = np.sqrt(sx ** 2 + sy ** 2).astype(np.float32, copy=False)

        grad_map = grad_cache[z_i]
        gy_lo = max(0, y_i - 2)
        gy_hi = min(shape[1], y_i + 3)
        gx_lo = max(0, x_i - 2)
        gx_hi = min(shape[2], x_i + 3)
        gradients[i] = float(grad_map[gy_lo:gy_hi, gx_lo:gx_hi].mean())

    return intensities, gradients


def _extract_local_gradient_magnitude(vol: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Compute Probe-C-aligned local gradient magnitude at coordinates."""
    _, gradients = _extract_local_intensity_and_gradient(vol, coords)
    return gradients


def _extract_with_keypoint_geometry(
    extractor: object,
    volume: np.ndarray,
    coords: np.ndarray,
    roi_mask: np.ndarray,
    max_kpts_per_slice: int = 500,
) -> np.ndarray:
    """Extract descriptors at random coords using SIFT keypoint geometry (size/angle).

    For each coordinate, we detect SIFT keypoints in the slice, find the nearest
    keypoint, and use its size for descriptor computation. Orientation is computed
    at the coordinate location using that size.

    This gives proper scale+rotation invariance at arbitrary sampling locations.

    Args:
        extractor: Must have detect_keypoints_2d() and compute_2d() methods
        volume: 3D volume (D, H, W)
        coords: Coordinates (N, 3) as (z, y, x)
        roi_mask: Boolean mask for valid regions
        max_kpts_per_slice: Max keypoints to detect per slice for size estimation

    Returns:
        Descriptors (N, D) with proper keypoint geometry
    """
    import cv2 as cv2_module

    if not hasattr(extractor, "detect_keypoints_2d") or not hasattr(extractor, "compute_2d"):
        raise ValueError("random_with_kp requires extractor with detect_keypoints_2d() and compute_2d()")

    n = len(coords)
    dim = int(getattr(extractor, "dim", 128))
    if n == 0:
        return np.zeros((0, dim), dtype=np.float32)

    descriptors = np.zeros((n, dim), dtype=np.float32)

    # Group coordinates by z-slice
    slice_groups: dict[int, list[tuple[int, int, int]]] = {}
    for idx, (z, y, x) in enumerate(coords):
        z_i = int(z)
        if z_i not in slice_groups:
            slice_groups[z_i] = []
        slice_groups[z_i].append((idx, int(y), int(x)))

    for z_i, points in slice_groups.items():
        if z_i < 0 or z_i >= volume.shape[0]:
            continue

        # Get 2D slice and mask
        slice_2d = volume[z_i]
        mask_2d = roi_mask[z_i] if roi_mask.ndim == 3 else None

        # Detect keypoints to get size distribution
        detected_kps = extractor.detect_keypoints_2d(
            slice_2d, mask=mask_2d, max_keypoints=max_kpts_per_slice
        )

        # Build KD-tree for nearest neighbor lookup if we have detected keypoints
        kp_coords = None
        kp_sizes = None
        if detected_kps:
            kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in detected_kps])
            kp_sizes = np.array([kp.size for kp in detected_kps])

        # Create keypoints at random locations with appropriate sizes
        keypoints = []
        indices = []
        for idx, y, x in points:
            # Find appropriate size from nearest detected keypoint
            if kp_coords is not None and len(kp_coords) > 0:
                dists = np.sqrt((kp_coords[:, 0] - x)**2 + (kp_coords[:, 1] - y)**2)
                nearest_idx = np.argmin(dists)
                size = float(kp_sizes[nearest_idx])
            else:
                size = _DEFAULT_KP_SIZE

            # Create keypoint with proper size, angle=-1 requests orientation computation
            kp = cv2_module.KeyPoint(float(x), float(y), size, -1)
            keypoints.append(kp)
            indices.append(idx)

        if not keypoints:
            continue

        # Compute descriptors - compute_2d handles orientation computation internally
        # if the extractor has compute_orientation=True
        descs = extractor.compute_2d(slice_2d, keypoints)
        for i, desc in zip(indices, descs):
            descriptors[i] = desc

    return descriptors


def _validate_paired_descriptors(
    d_a_raw: np.ndarray,
    d_b_raw: np.ndarray,
    min_desc_norm: float,
) -> np.ndarray:
    """Return boolean mask of valid paired descriptors (row-aligned)."""
    if d_a_raw.ndim != 2 or d_b_raw.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got d_a_raw.ndim={d_a_raw.ndim}, d_b_raw.ndim={d_b_raw.ndim}")
    if d_a_raw.shape[0] != d_b_raw.shape[0]:
        raise ValueError(f"Paired descriptors must align by row: {d_a_raw.shape[0]} != {d_b_raw.shape[0]}")

    n = d_a_raw.shape[0]
    valid = np.ones(n, dtype=bool)
    if n == 0:
        return valid

    norms_a = np.linalg.norm(d_a_raw, axis=1)
    norms_b = np.linalg.norm(d_b_raw, axis=1)
    valid &= np.isfinite(norms_a) & (norms_a > min_desc_norm)
    valid &= np.isfinite(norms_b) & (norms_b > min_desc_norm)
    return valid


def _extract_paired_from_keypoints(
    extractor: object,
    vol_a: np.ndarray,
    vol_b: np.ndarray,
    kps_with_z: list[tuple[int, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract paired descriptors from both volumes using shared 2D keypoint geometry.

    Args:
        extractor: Must implement `compute_2d(image2d, keypoints) -> (N, D)`.
        vol_a: Volume A, shape (D, H, W)
        vol_b: Volume B, shape (D, H, W)
        kps_with_z: List of (z, cv2.KeyPoint) tuples

    Returns:
        (descs_a, descs_b): Two arrays of shape (N, D), aligned by keypoint index.
    """
    if not hasattr(extractor, "compute_2d"):
        raise ValueError("_extract_paired_from_keypoints requires extractor.compute_2d(image, keypoints).")

    n = len(kps_with_z)
    dim = int(getattr(extractor, "dim", 128))
    if n == 0:
        empty = np.zeros((0, dim), dtype=np.float32)
        return empty, empty

    descs_a = np.zeros((n, dim), dtype=np.float32)
    descs_b = np.zeros((n, dim), dtype=np.float32)

    # Batch by slice for performance while preserving global order.
    groups: dict[int, list[tuple[int, object]]] = {}
    for idx, (z, kp) in enumerate(kps_with_z):
        z_i = int(z)
        if z_i not in groups:
            groups[z_i] = []
        groups[z_i].append((idx, kp))

    for z_i, items in groups.items():
        if z_i < 0 or z_i >= vol_a.shape[0] or z_i >= vol_b.shape[0]:
            continue
        indices = [idx for idx, _ in items]
        kps = [kp for _, kp in items]
        d_a = extractor.compute_2d(vol_a[z_i], kps).astype(np.float32, copy=False)
        d_b = extractor.compute_2d(vol_b[z_i], kps).astype(np.float32, copy=False)
        if d_a.shape != (len(indices), dim) or d_b.shape != (len(indices), dim):
            raise ValueError(
                f"compute_2d returned unexpected shapes: d_a={d_a.shape}, d_b={d_b.shape}, expected ({len(indices)}, {dim})"
            )
        descs_a[indices] = d_a
        descs_b[indices] = d_b

    return descs_a, descs_b


def _extract_paired_from_keypoints_sets(
    extractor: object,
    vol_a: np.ndarray,
    vol_b: np.ndarray,
    kps_a_with_z: list[tuple[int, object]],
    kps_b_with_z: list[tuple[int, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract paired descriptors using two aligned keypoint lists (A and B may differ).

    Args:
        extractor: Must implement `compute_2d(image2d, keypoints) -> (N, D)`.
        vol_a: Volume A, shape (D, H, W)
        vol_b: Volume B, shape (D, H, W)
        kps_a_with_z: List of (z, cv2.KeyPoint) in A space
        kps_b_with_z: List of (z, cv2.KeyPoint) in B space; must align by index with kps_a_with_z

    Returns:
        (descs_a, descs_b): Two arrays of shape (N, D), aligned by keypoint index.
    """
    if len(kps_a_with_z) != len(kps_b_with_z):
        raise ValueError(
            f"_extract_paired_from_keypoints_sets requires aligned lists; got {len(kps_a_with_z)} != {len(kps_b_with_z)}"
        )
    if not hasattr(extractor, "compute_2d"):
        raise ValueError("_extract_paired_from_keypoints_sets requires extractor.compute_2d(image, keypoints).")

    n = len(kps_a_with_z)
    dim = int(getattr(extractor, "dim", 128))
    if n == 0:
        empty = np.zeros((0, dim), dtype=np.float32)
        return empty, empty

    descs_a = np.zeros((n, dim), dtype=np.float32)
    descs_b = np.zeros((n, dim), dtype=np.float32)

    groups_a: dict[int, list[tuple[int, object]]] = {}
    groups_b: dict[int, list[tuple[int, object]]] = {}

    for idx, (z, kp) in enumerate(kps_a_with_z):
        z_i = int(z)
        groups_a.setdefault(z_i, []).append((idx, kp))
    for idx, (z, kp) in enumerate(kps_b_with_z):
        z_i = int(z)
        groups_b.setdefault(z_i, []).append((idx, kp))

    for z_i, items in groups_a.items():
        if z_i < 0 or z_i >= vol_a.shape[0]:
            continue
        indices = [idx for idx, _ in items]
        kps = [kp for _, kp in items]
        d = extractor.compute_2d(vol_a[z_i], kps).astype(np.float32, copy=False)
        if d.shape != (len(indices), dim):
            raise ValueError(
                f"compute_2d returned unexpected shape for A: d={d.shape}, expected ({len(indices)}, {dim})"
            )
        descs_a[indices] = d

    for z_i, items in groups_b.items():
        if z_i < 0 or z_i >= vol_b.shape[0]:
            continue
        indices = [idx for idx, _ in items]
        kps = [kp for _, kp in items]
        d = extractor.compute_2d(vol_b[z_i], kps).astype(np.float32, copy=False)
        if d.shape != (len(indices), dim):
            raise ValueError(
                f"compute_2d returned unexpected shape for B: d={d.shape}, expected ({len(indices)}, {dim})"
            )
        descs_b[indices] = d

    return descs_a, descs_b


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_paired_descriptors(
    cases: List[CaseData],
    modalities: List[str],
    extractor: object,
    n_samples_per_case: int = 5000,
    seed: int = 42,
    aug_rotate_deg: float = 0.0,
    aug_translate_mm: float = 0.0,
    aug_scale: float = 0.0,
    aug_shear_deg: float = 0.0,
    n_augmentations: int = 1,
    crop_ratio: float = 1.0,
    asymmetric_aug: bool = False,
    aug_2d_only: bool = True,
    min_valid_voxels: int = 100,
    min_valid_pairs: int = 50,
    min_desc_norm: float = 0.1,
    pair_sampling: str = "random",
    max_kpts_per_slice: int = 500,
    show_progress: bool = True,
    return_local_props: bool = False,
    num_workers: int = 1,
    max_samples: int | None = None,
) -> tuple:
    """Extract descriptor pairs at SAME locations (for aligned data).

    For each patient:
    - Sample N locations (optionally from a random crop/region)
    - Optionally apply random geometric augmentation
    - Extract d_mod_a AND d_mod_b at those locations
    - This gives us paired training data

    Two augmentation modes:
    1. Symmetric (default): Apply SAME transform to BOTH modalities.
       This preserves location correspondence while teaching rotation/translation invariance.

    2. Asymmetric (asymmetric_aug=True): Keep modality A at canonical coords,
       apply transform only to modality B. This teaches the model that the two
       descriptors belong together even when sampling locations differ slightly.
       Better for learning misregistration robustness.

    Args:
        cases: List of CaseData objects
        modalities: List of modality names
        extractor: Descriptor extractor instance
        n_samples_per_case: Number of samples per case
        seed: Random seed
        aug_rotate_deg: Max rotation for augmentation
        aug_translate_mm: Max translation for augmentation
        n_augmentations: Number of augmentation passes per case
        crop_ratio: Fraction of volume to sample from (0.7 = central 70%)
        asymmetric_aug: If True, only augment the second modality
        aug_2d_only: If True (default), constrain transforms to axial in-plane (z fixed).
            Set to False for 3D rotations (not recommended for 2D SIFT descriptors).
        min_valid_voxels: Minimum number of candidate voxels required per pass.
            If fewer are available, the pass is skipped.
        min_valid_pairs: Minimum number of valid descriptor pairs required per pass
            after validity filtering. If fewer are available, the pass is skipped.
        min_desc_norm: Minimum L2 norm for a descriptor to be considered valid.
            This is intentionally configurable because different descriptor families
            (SIFT, CNN, ViT) can have very different norm distributions.
        pair_sampling: Coordinate sampling strategy. "random" samples from the
            joint ROI mask. "sift_union" detects SIFT keypoints in both modalities
            within the same ROI, takes half from each, unions the coords, then
            fills any remainder with random ROI samples. "2d_sift_ref" detects 2D SIFT
            keypoints slice-by-slice in modality A and computes paired descriptors in
            both modalities using the same keypoint geometry (x, y, size, angle).
            "2d_sift_dual" does the same but samples keypoints from both modalities
            (half from each) and concatenates paired descriptors.
        show_progress: Whether to show tqdm progress bar (default True).
        return_local_props: If True, also return local property arrays aligned
            with each descriptor row. Includes:
            - local_props["gradient"][modality]: Probe-C gradient magnitude labels
            - local_props["intensity"][modality]: Probe-B local intensity labels
            - local_props["coords"][modality]: Normalized [z, y, x] coordinates in [0, 1]
        num_workers: Number of threads for parallel case processing.
            Defaults to 1 (sequential). Values > 1 require the extractor to
            implement a ``clone()`` method that returns an independent copy.

            Thread-safety requirements for ``clone()``:

            - Must return a NEW instance (not ``self``).
            - Clones may share **read-only** state (e.g., model weights, config).
            - Clones must have **separate mutable state** (buffers, caches, OpenCV
              objects like ``cv2.SIFT_create()``).
            - After cloning, ``extract()`` / ``compute_2d()`` will be called
              concurrently across clones and must not share mutable state.
        max_samples: If set, stop accumulating after this many descriptor pairs
            and trim the output to exactly this size. Useful for capping memory
            usage during quick validation runs. ``None`` (default) means no limit.

    Returns:
        If return_local_props=False:
            (descriptors dict, total sample count)
        If return_local_props=True:
            (descriptors dict, total sample count, local_props dict)
    """
    if len(modalities) != 2:
        raise ValueError(
            f"extract_paired_descriptors expects exactly 2 modalities, got {modalities}."
        )
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if min_valid_voxels <= 0:
        raise ValueError(f"min_valid_voxels must be > 0, got {min_valid_voxels}")
    if min_valid_pairs <= 0:
        raise ValueError(f"min_valid_pairs must be > 0, got {min_valid_pairs}")
    if min_desc_norm < 0:
        raise ValueError(f"min_desc_norm must be >= 0, got {min_desc_norm}")
    if pair_sampling not in _PAIR_SAMPLING_MODES:
        raise ValueError(
            f"pair_sampling must be one of {sorted(_PAIR_SAMPLING_MODES)}, got {pair_sampling!r}"
        )
    max_kpts_per_slice = int(max_kpts_per_slice)
    if max_kpts_per_slice <= 0:
        raise ValueError(f"max_kpts_per_slice must be > 0, got {max_kpts_per_slice}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be > 0 when set, got {max_samples}")
    if pair_sampling in {"2d_sift_ref", "2d_sift_dual"}:
        if (aug_scale > 0 or aug_shear_deg > 0) and not aug_2d_only:
            raise ValueError(
                f"pair_sampling={pair_sampling!r} with affine augmentation requires aug_2d_only=True "
                f"(out-of-plane affine not supported)."
            )
        if asymmetric_aug and not aug_2d_only:
            raise ValueError(
                f"pair_sampling={pair_sampling!r} with asymmetric_aug requires aug_2d_only=True "
                f"(keypoint warping across slices not supported)."
            )
        if not hasattr(extractor, "detect_keypoints_2d") or not hasattr(extractor, "compute_2d"):
            raise ValueError(
                f"pair_sampling={pair_sampling!r} requires extractor.detect_keypoints_2d() and extractor.compute_2d()."
            )
    if pair_sampling == "random_with_kp":
        if not hasattr(extractor, "detect_keypoints_2d") or not hasattr(extractor, "compute_2d"):
            raise ValueError(
                f"pair_sampling={pair_sampling!r} requires extractor.detect_keypoints_2d() and extractor.compute_2d()."
            )

    empty_dim = int(getattr(extractor, "dim", 384))
    use_augmentation = aug_rotate_deg > 0 or aug_translate_mm > 0 or aug_scale > 0 or aug_shear_deg > 0
    use_crop = crop_ratio < 1.0

    # --- Parallelism setup ---
    effective_workers = min(num_workers, len(cases)) if cases else 1
    effective_workers = _prepare_parallel_extractor(extractor, effective_workers, logger)

    # Thread-local storage for cloned extractors — defined here (not module-level)
    # so cloned extractors are GC'd when this function returns.
    _tls = threading.local()

    # --- Per-case processing function ---
    def _process_case(case_idx: int, case: CaseData) -> _CaseResult:
        # Get thread-local extractor
        if effective_workers > 1:
            if not hasattr(_tls, "extractor"):
                try:
                    _tls.extractor = extractor.clone()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to clone {type(extractor).__name__} in worker thread "
                        f"(case_idx={case_idx})"
                    ) from e
            ext = _tls.extractor
        else:
            ext = extractor

        result = _CaseResult()

        vol_a_ref = case.volumes[modalities[0]]
        vol_b_ref = case.volumes[modalities[1]]
        shape = np.array(vol_a_ref.shape)
        center = (shape - 1.0) / 2.0
        base_margin = 15

        affine = case.affines.get(modalities[0])
        voxel_sizes_mm = None
        if affine is not None:
            voxel_sizes_mm = np.sqrt(np.sum(np.square(affine[:3, :3]), axis=0))

        # Number of augmentation passes per case
        n_passes = n_augmentations if (use_augmentation or use_crop) else 1

        for aug_idx in range(n_passes):
            rng = np.random.RandomState(seed + case_idx * 1000 + aug_idx)

            # Compute margin for this pass
            safe_crop_min, safe_crop_max = compute_safe_crop_bounds(
                tuple(int(s) for s in shape), base_margin
            )
            if use_crop:
                crop_margin = ((1.0 - crop_ratio) / 2.0) * shape
                offset = rng.uniform(-0.1, 0.1, size=3) * shape
                crop_min = (crop_margin + offset).astype(int).clip(safe_crop_min, None)
                crop_max = (shape - crop_margin + offset).astype(int).clip(None, safe_crop_max)
                crop_min = np.maximum(crop_min, safe_crop_min)
                crop_max = np.minimum(crop_max, safe_crop_max).astype(int)
            else:
                crop_min = safe_crop_min
                crop_max = safe_crop_max

            # Generate transform and optionally warp volumes
            lin_mat = None
            trans_vox = None
            if use_augmentation:
                if (aug_scale > 0 or aug_shear_deg > 0) and not aug_2d_only:
                    raise ValueError(
                        "Affine augmentation requires aug_2d_only=True (out-of-plane affine not supported)."
                    )
                if aug_2d_only and (aug_scale > 0 or aug_shear_deg > 0):
                    lin_mat, trans_vox = generate_random_affine_in_plane(
                        rotate_deg=aug_rotate_deg,
                        translate_mm=aug_translate_mm,
                        scale=aug_scale,
                        shear_deg=aug_shear_deg,
                        center=center,
                        rng=rng,
                        voxel_sizes_mm=voxel_sizes_mm,
                        snap_translation_to_vox=asymmetric_aug,
                        fixed_axis=0,
                    )
                else:
                    gen = generate_random_transform_in_plane if aug_2d_only else generate_random_transform
                    gen_kwargs: dict = {
                        "rotate_deg": aug_rotate_deg,
                        "translate_mm": aug_translate_mm,
                        "center": center,
                        "rng": rng,
                        "voxel_sizes_mm": voxel_sizes_mm,
                        "snap_translation_to_vox": asymmetric_aug,
                    }
                    if aug_2d_only:
                        gen_kwargs["fixed_axis"] = 0
                    lin_mat, trans_vox = gen(**gen_kwargs)

                warp_fn = warp_volume_affine if (aug_scale > 0 or aug_shear_deg > 0) else warp_volume_rigid
                if asymmetric_aug:
                    vol_a = vol_a_ref
                    vol_b = warp_fn(vol_b_ref, lin_mat, trans_vox, center=center, order=1, cval=0.0)
                else:
                    vol_a = warp_fn(vol_a_ref, lin_mat, trans_vox, center=center, order=1, cval=0.0)
                    vol_b = warp_fn(vol_b_ref, lin_mat, trans_vox, center=center, order=1, cval=0.0)
            else:
                vol_a = vol_a_ref
                vol_b = vol_b_ref

            # Joint mask: only where BOTH modalities have valid data
            if use_augmentation and asymmetric_aug:
                mask_a = np.abs(vol_a_ref) > 0.01 * np.max(np.abs(vol_a_ref))
                mask_b = np.abs(vol_b_ref) > 0.01 * np.max(np.abs(vol_b_ref))
            else:
                mask_a = np.abs(vol_a) > 0.01 * np.max(np.abs(vol_a))
                mask_b = np.abs(vol_b) > 0.01 * np.max(np.abs(vol_b))
            mask = mask_a & mask_b

            # Erode to avoid FOV boundaries (depth-safe)
            erode_margin = 5
            mask = safe_preprocess_roi(mask, erode_iterations=erode_margin, base_margin=base_margin)

            z, y, x = np.where(mask)

            # Filter to crop region
            in_crop = (
                (z >= crop_min[0]) & (z < crop_max[0]) &
                (y >= crop_min[1]) & (y < crop_max[1]) &
                (x >= crop_min[2]) & (x < crop_max[2])
            )
            z, y, x = z[in_crop], y[in_crop], x[in_crop]

            if len(z) < min_valid_voxels:
                result.skipped_low_mask += 1
                continue

            coords_pool = np.stack([z, y, x], axis=1).astype(np.int32, copy=False)
            n_goal = int(min(n_samples_per_case, len(coords_pool)))
            if pair_sampling in {"2d_sift_ref", "2d_sift_dual"}:
                keep_upright = bool(getattr(ext, "compute_orientation", True)) is False
                if use_augmentation and asymmetric_aug:
                    vol_a_det = vol_a_ref
                    vol_b_det = vol_b_ref
                else:
                    vol_a_det = vol_a
                    vol_b_det = vol_b

                def _extract_from_anchor_keypoints(
                    kps_anchor: list[tuple[int, object]],
                    _ext: object = ext,
                    _vol_a: np.ndarray = vol_a,
                    _vol_b: np.ndarray = vol_b,
                    _lin_mat: object = lin_mat,
                    _trans_vox: object = trans_vox,
                    _center: np.ndarray = center,
                    _shape: np.ndarray = shape,
                    _keep_upright: bool = keep_upright,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                    coords_a_key = _coords_from_keypoints_with_z(kps_anchor, _shape)
                    if use_augmentation and asymmetric_aug:
                        kps_anchor_b = transform_keypoints_with_z(
                            kps_anchor,
                            _lin_mat,
                            _trans_vox,
                            _center,
                            keep_upright=_keep_upright,
                            update_size=(aug_scale > 0 or aug_shear_deg > 0),
                            allow_z_change=False,
                        )
                        coords_b_key = _coords_from_keypoints_with_z(kps_anchor_b, _shape)
                        d_a_key, d_b_key = _extract_paired_from_keypoints_sets(
                            _ext, _vol_a, _vol_b, kps_anchor, kps_anchor_b
                        )
                        return d_a_key, d_b_key, coords_a_key, coords_b_key
                    d_a_key, d_b_key = _extract_paired_from_keypoints(_ext, _vol_a, _vol_b, kps_anchor)
                    return d_a_key, d_b_key, coords_a_key, coords_a_key

                if pair_sampling == "2d_sift_ref":
                    kps_with_z = sample_keypoints_2d_ref(
                        ext,
                        vol_ref=vol_a_det,
                        roi_mask=mask,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        n_goal=n_goal,
                        max_keypoints_per_slice=max_kpts_per_slice,
                        rng=rng,
                    )
                    d_a_raw, d_b_raw, coords_a_raw, coords_b_raw = _extract_from_anchor_keypoints(kps_with_z)
                else:
                    kps_a_anchor, kps_b_anchor = sample_keypoints_2d_dual(
                        ext,
                        vol_a=vol_a_det,
                        vol_b=vol_b_det,
                        roi_mask=mask,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        n_goal=n_goal,
                        max_keypoints_per_slice=max_kpts_per_slice,
                        rng=rng,
                    )
                    d_a1, d_b1, c_a1, c_b1 = _extract_from_anchor_keypoints(kps_a_anchor)
                    d_a2, d_b2, c_a2, c_b2 = _extract_from_anchor_keypoints(kps_b_anchor)
                    parts_a = [p for p in (d_a1, d_a2) if p.size != 0]
                    parts_b = [p for p in (d_b1, d_b2) if p.size != 0]
                    parts_c_a = [p for p in (c_a1, c_a2) if p.size != 0]
                    parts_c_b = [p for p in (c_b1, c_b2) if p.size != 0]
                    if parts_a:
                        d_a_raw = np.vstack(parts_a)
                        d_b_raw = np.vstack(parts_b)
                        coords_a_raw = np.vstack(parts_c_a)
                        coords_b_raw = np.vstack(parts_c_b)
                    else:
                        d_a_raw = np.zeros((0, empty_dim), dtype=np.float32)
                        d_b_raw = np.zeros((0, empty_dim), dtype=np.float32)
                        coords_a_raw = np.zeros((0, 3), dtype=np.int32)
                        coords_b_raw = np.zeros((0, 3), dtype=np.int32)

                valid = _validate_paired_descriptors(d_a_raw, d_b_raw, min_desc_norm)

                if int(valid.sum()) < min_valid_pairs:
                    result.skipped_low_valid += 1
                    continue

                result.descs_a.append(d_a_raw[valid])
                result.descs_b.append(d_b_raw[valid])
                if return_local_props:
                    valid_a = coords_a_raw[valid]
                    valid_b = coords_b_raw[valid]
                    int_a, grad_a = _extract_local_intensity_and_gradient(vol_a, valid_a)
                    int_b, grad_b = _extract_local_intensity_and_gradient(vol_b, valid_b)
                    result.local_intensities_a.append(int_a)
                    result.local_intensities_b.append(int_b)
                    result.local_gradients_a.append(grad_a)
                    result.local_gradients_b.append(grad_b)
                    shape_f = (shape - 1).astype(np.float32)
                    norm_a = valid_a.astype(np.float32) / np.maximum(shape_f, 1.0)
                    norm_b = valid_b.astype(np.float32) / np.maximum(shape_f, 1.0)
                    result.local_coords_a.append(norm_a)
                    result.local_coords_b.append(norm_b)
                continue

            if pair_sampling == "sift_union":
                if use_augmentation and asymmetric_aug:
                    vol_a_det = vol_a_ref
                    vol_b_det = vol_b_ref
                else:
                    vol_a_det = vol_a
                    vol_b_det = vol_b
                coords = sample_coords_sift_union_from_roi(
                    ext,
                    vol_a=vol_a_det,
                    vol_b=vol_b_det,
                    roi_mask=mask,
                    crop_min=crop_min,
                    crop_max=crop_max,
                    coords_pool=coords_pool,
                    n_goal=n_goal,
                    rng=rng,
                )
            else:
                idx = rng.choice(len(coords_pool), size=n_goal, replace=False)
                coords = coords_pool[idx]

            # Build paired coordinates
            if use_augmentation and asymmetric_aug:
                coords_a = coords.astype(np.int32)
                coords_b_f = apply_transform_to_coords(coords_a, lin_mat, trans_vox, center)
                coords_b = np.round(coords_b_f).astype(np.int32)

                valid_transform = np.all(
                    (coords_b >= safe_crop_min) & (coords_b < safe_crop_max),
                    axis=1,
                )
                if int(valid_transform.sum()) < min_valid_voxels:
                    result.skipped_low_transform += 1
                    continue

                mask_b_aug = np.abs(vol_b) > 0.01 * np.max(np.abs(vol_b))
                mask_b_aug = safe_preprocess_roi(mask_b_aug, erode_iterations=erode_margin, base_margin=base_margin)
                valid_indices = np.where(valid_transform)[0]
                coords_b_valid = coords_b[valid_indices]
                mask_check = mask_b_aug[coords_b_valid[:, 0], coords_b_valid[:, 1], coords_b_valid[:, 2]]
                valid_transform[valid_indices[~mask_check]] = False

                coords_a = coords_a[valid_transform]
                coords_b = coords_b[valid_transform]
            else:
                coords_a = coords.astype(np.int32)
                coords_b = coords_a

            # Extract descriptors (using thread-local ext)
            descs: dict[str, np.ndarray] = {}
            if pair_sampling == "random_with_kp":
                descs[modalities[0]] = _extract_with_keypoint_geometry(
                    ext, vol_a, coords_a, mask, max_kpts_per_slice
                ).astype(np.float32)
                descs[modalities[1]] = _extract_with_keypoint_geometry(
                    ext, vol_b, coords_b, mask, max_kpts_per_slice
                ).astype(np.float32)
            else:
                descs[modalities[0]] = ext.extract(
                    vol_a, coords_a
                ).astype(np.float32)
                descs[modalities[1]] = ext.extract(
                    vol_b, coords_b
                ).astype(np.float32)

            # Filter: keep only where BOTH descriptors are valid
            valid = np.ones(len(descs[modalities[0]]), dtype=bool)
            for mod in modalities:
                norms = np.linalg.norm(descs[mod], axis=1)
                valid &= np.isfinite(norms) & (norms > min_desc_norm)

            if int(valid.sum()) < min_valid_pairs:
                result.skipped_low_valid += 1
                continue

            result.descs_a.append(descs[modalities[0]][valid])
            result.descs_b.append(descs[modalities[1]][valid])
            if return_local_props:
                int_a, grad_a = _extract_local_intensity_and_gradient(vol_a, coords_a[valid])
                int_b, grad_b = _extract_local_intensity_and_gradient(vol_b, coords_b[valid])
                result.local_intensities_a.append(int_a)
                result.local_intensities_b.append(int_b)
                result.local_gradients_a.append(grad_a)
                result.local_gradients_b.append(grad_b)
                shape_f = (shape - 1).astype(np.float32)
                norm_a = coords_a[valid].astype(np.float32) / np.maximum(shape_f, 1.0)
                norm_b = coords_b[valid].astype(np.float32) / np.maximum(shape_f, 1.0)
                result.local_coords_a.append(norm_a)
                result.local_coords_b.append(norm_b)

        return result

    # --- Dispatch ---
    desc_msg = "Extracting paired descriptors"
    if use_augmentation:
        aug_mode = "asymmetric" if asymmetric_aug else "symmetric"
        if aug_2d_only:
            aug_mode += "+2d"
        affine_str = ""
        if aug_scale > 0 or aug_shear_deg > 0:
            affine_str = f", scale±{aug_scale:.2f}, shear±{aug_shear_deg:.1f}°"
        desc_msg += (
            f" (aug: ±{aug_rotate_deg}°, ±{aug_translate_mm}mm{affine_str}, x{n_augmentations}, {aug_mode})"
        )
    if use_crop:
        desc_msg += f" (crop: {crop_ratio:.0%})"

    if effective_workers > 1:
        from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed

        case_results: list[_CaseResult | None] = [None] * len(cases)
        pbar = tqdm(total=len(cases), desc=desc_msg) if show_progress else None
        _accumulated = 0
        _enough = False
        try:
            # Limit OpenCV to 1 internal thread to avoid oversubscription:
            # with N Python worker threads each spawning M OpenCV threads,
            # we'd have N*M threads competing for CPU cores.
            with _opencv_thread_limit(1):
                with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                    futures = {pool.submit(_process_case, i, c): i for i, c in enumerate(cases)}
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            case_results[idx] = future.result()
                        except CancelledError:
                            continue  # slot stays None
                        if pbar:
                            pbar.update(1)
                        # Early stopping: cancel queued futures once we
                        # have enough samples.  In-flight work finishes
                        # naturally; only queued (not-yet-started) futures
                        # are actually cancelled.  We do NOT break here —
                        # the loop drains so that all non-cancelled results
                        # are collected, keeping the reduce phase
                        # deterministic (it processes slots in index order).
                        if max_samples is not None and not _enough:
                            cr = case_results[idx]
                            _accumulated += sum(d.shape[0] for d in cr.descs_a)
                            if _accumulated >= max_samples:
                                _enough = True
                                n_cancelled = 0
                                for f in futures:
                                    if f.cancel():
                                        n_cancelled += 1
                                if n_cancelled > 0:
                                    logger.info(
                                        "max_samples=%d reached (%d accumulated);"
                                        " cancelled %d pending futures",
                                        max_samples, _accumulated, n_cancelled,
                                    )
        finally:
            if pbar:
                pbar.close()
    else:
        case_iter = tqdm(enumerate(cases), total=len(cases), desc=desc_msg) if show_progress else enumerate(cases)
        case_results = []
        _accumulated = 0
        for i, c in case_iter:
            cr = _process_case(i, c)
            case_results.append(cr)
            if max_samples is not None:
                _accumulated += sum(d.shape[0] for d in cr.descs_a)
                if _accumulated >= max_samples:
                    if show_progress and hasattr(case_iter, "close"):
                        case_iter.close()
                    break

    # --- Reduce results in case order ---
    all_descriptors: dict[str, list[np.ndarray]] = {mod: [] for mod in modalities}
    all_local_gradients: dict[str, list[np.ndarray]] | None = {mod: [] for mod in modalities} if return_local_props else None
    all_local_intensities: dict[str, list[np.ndarray]] | None = {mod: [] for mod in modalities} if return_local_props else None
    all_local_coords: dict[str, list[np.ndarray]] | None = {mod: [] for mod in modalities} if return_local_props else None

    skipped_low_mask = 0
    skipped_low_transform = 0
    skipped_low_valid = 0

    _reduce_accumulated = 0
    for cr in case_results:
        if cr is None:
            continue  # cancelled future (early stopping with workers)
        skipped_low_mask += cr.skipped_low_mask
        skipped_low_transform += cr.skipped_low_transform
        skipped_low_valid += cr.skipped_low_valid
        all_descriptors[modalities[0]].extend(cr.descs_a)
        all_descriptors[modalities[1]].extend(cr.descs_b)
        if return_local_props:
            all_local_gradients[modalities[0]].extend(cr.local_gradients_a)
            all_local_gradients[modalities[1]].extend(cr.local_gradients_b)
            all_local_intensities[modalities[0]].extend(cr.local_intensities_a)
            all_local_intensities[modalities[1]].extend(cr.local_intensities_b)
            all_local_coords[modalities[0]].extend(cr.local_coords_a)
            all_local_coords[modalities[1]].extend(cr.local_coords_b)
        if max_samples is not None:
            _reduce_accumulated += sum(d.shape[0] for d in cr.descs_a)
            if _reduce_accumulated >= max_samples:
                break

    # Concatenate
    for mod in modalities:
        if all_descriptors[mod]:
            all_descriptors[mod] = np.vstack(all_descriptors[mod])
        else:
            all_descriptors[mod] = np.array([], dtype=np.float32).reshape(0, empty_dim)

    local_props = None
    if return_local_props:
        gradients_by_mod: dict[str, np.ndarray] = {}
        intensities_by_mod: dict[str, np.ndarray] = {}
        coords_by_mod: dict[str, np.ndarray] = {}
        for mod in modalities:
            if all_local_gradients[mod]:
                gradients = np.concatenate(all_local_gradients[mod]).astype(np.float32, copy=False)
            else:
                gradients = np.zeros((0,), dtype=np.float32)
            if gradients.shape[0] != all_descriptors[mod].shape[0]:
                raise ValueError(
                    f"Gradient labels misaligned for {mod}: gradients={gradients.shape[0]} "
                    f"descriptors={all_descriptors[mod].shape[0]}"
                )
            gradients_by_mod[mod] = gradients
            if all_local_intensities[mod]:
                intensities = np.concatenate(all_local_intensities[mod]).astype(np.float32, copy=False)
            else:
                intensities = np.zeros((0,), dtype=np.float32)
            if intensities.shape[0] != all_descriptors[mod].shape[0]:
                raise ValueError(
                    f"Intensity labels misaligned for {mod}: intensities={intensities.shape[0]} "
                    f"descriptors={all_descriptors[mod].shape[0]}"
                )
            intensities_by_mod[mod] = intensities
            if all_local_coords[mod]:
                coords_arr = np.vstack(all_local_coords[mod]).astype(np.float32, copy=False)
            else:
                coords_arr = np.zeros((0, 3), dtype=np.float32)
            if coords_arr.shape[0] != all_descriptors[mod].shape[0]:
                raise ValueError(
                    f"Coordinate labels misaligned for {mod}: coords={coords_arr.shape[0]} "
                    f"descriptors={all_descriptors[mod].shape[0]}"
                )
            coords_by_mod[mod] = coords_arr
        local_props = {
            "gradient": gradients_by_mod,
            "intensity": intensities_by_mod,
            "coords": coords_by_mod,
        }

    # --- Final trim to max_samples ---
    if max_samples is not None and len(all_descriptors[modalities[0]]) > max_samples:
        for mod in modalities:
            all_descriptors[mod] = all_descriptors[mod][:max_samples]
        if local_props is not None:
            for prop_key in local_props:
                for mod in local_props[prop_key]:
                    local_props[prop_key][mod] = local_props[prop_key][mod][:max_samples]

    n_samples = len(all_descriptors[modalities[0]])
    if skipped_low_mask or skipped_low_transform or skipped_low_valid:
        logger.warning(
            "extract_paired_descriptors skipped passes due to low-signal or low-validity: "
            "low_mask=%d low_transform=%d low_valid=%d (cases=%d, passes=%d)",
            skipped_low_mask,
            skipped_low_transform,
            skipped_low_valid,
            len(cases),
            len(cases) * (n_augmentations if (use_augmentation or use_crop) else 1),
        )
    if return_local_props:
        return all_descriptors, n_samples, local_props
    return all_descriptors, n_samples


def extract_paired_descriptors_per_case(
    cases: List[CaseData],
    modalities: List[str],
    extractor: object,
    n_samples_per_case: int = 5000,
    seed: int = 42,
    pair_sampling: str = "random",
    max_kpts_per_slice: int = 500,
    show_progress: bool = True,
    num_workers: int = 1,
) -> List[Tuple[Dict[str, np.ndarray], int]]:
    """Extract paired descriptors for each case separately, returning per-case results.

    This is useful for evaluation where per-case metrics are needed.
    Unlike extract_paired_descriptors which pools all cases, this returns
    a list of (descriptors_dict, n_samples) tuples, one per case.

    Args:
        cases: List of CaseData objects
        modalities: List of exactly 2 modality names
        extractor: Descriptor extractor instance
        n_samples_per_case: Number of samples per case
        seed: Random seed
        pair_sampling: Coordinate sampling strategy
        max_kpts_per_slice: Max keypoints per slice for keypoint-based sampling
        show_progress: Whether to show progress bar
        num_workers: Number of threads for parallel case processing.
            Defaults to 1 (sequential). Values > 1 require the extractor to
            implement a ``clone()`` method that returns an independent copy.
            See ``extract_paired_descriptors`` for the full thread-safety contract.

    Returns:
        List of (descriptors_dict, n_samples) tuples, one per case
    """
    if len(modalities) != 2:
        raise ValueError(f"Expected exactly 2 modalities, got {modalities}")
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    effective_workers = min(num_workers, len(cases)) if cases else 1
    effective_workers = _prepare_parallel_extractor(extractor, effective_workers, logger)

    # Thread-local storage for cloned extractors (one clone per worker thread,
    # not per case) — matches the pattern in extract_paired_descriptors.
    _tls = threading.local()

    def _extract_one(case_idx: int, case: CaseData) -> tuple[Dict[str, np.ndarray], int]:
        if effective_workers > 1:
            if not hasattr(_tls, "extractor"):
                _tls.extractor = extractor.clone()
            ext = _tls.extractor
        else:
            ext = extractor
        desc, n = extract_paired_descriptors(
            [case],
            modalities,
            ext,
            n_samples_per_case=n_samples_per_case,
            seed=seed + case_idx,
            pair_sampling=pair_sampling,
            max_kpts_per_slice=max_kpts_per_slice,
            show_progress=False,
            num_workers=1,
        )
        return desc, n

    if effective_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[tuple[Dict[str, np.ndarray], int] | None] = [None] * len(cases)
        pbar = tqdm(total=len(cases), desc="Extracting descriptors") if show_progress else None
        try:
            # Limit OpenCV to 1 internal thread to avoid oversubscription.
            with _opencv_thread_limit(1):
                with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                    futures = {pool.submit(_extract_one, i, c): i for i, c in enumerate(cases)}
                    for future in as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()
                        if pbar:
                            pbar.update(1)
        finally:
            if pbar:
                pbar.close()
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise RuntimeError(f"Cases at indices {missing} were not processed")
        return results
    else:
        results_seq: list[tuple[Dict[str, np.ndarray], int]] = []
        iterator = tqdm(cases, desc="Extracting descriptors") if show_progress else cases
        for case_idx, case in enumerate(iterator):
            desc, n = extract_paired_descriptors(
                [case],
                modalities,
                extractor,
                n_samples_per_case=n_samples_per_case,
                seed=seed + case_idx,
                pair_sampling=pair_sampling,
                max_kpts_per_slice=max_kpts_per_slice,
                show_progress=False,
                num_workers=1,
            )
            results_seq.append((desc, n))
        return results_seq
