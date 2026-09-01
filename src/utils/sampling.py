"""Coordinate sampling utilities for descriptor extraction."""

import numpy as np
from typing import Any, Sequence, Union

from src.utils.roi_preprocessing import safe_preprocess_roi


def ordered_unique_ints(values: Union[Sequence[int], np.ndarray]) -> list[int]:
    """Return unique integers preserving first-occurrence order.

    Args:
        values: Sequence of integers (list, tuple, or numpy array).

    Returns:
        List of unique integers in the order they first appeared.

    Examples:
        >>> ordered_unique_ints([3, 1, 2, 1, 3, 4])
        [3, 1, 2, 4]
        >>> ordered_unique_ints(np.array([5, 5, 3, 3, 1]))
        [5, 3, 1]
    """
    seen: set[int] = set()
    out: list[int] = []
    for v in values:
        vv = int(v)
        if vv in seen:
            continue
        seen.add(vv)
        out.append(vv)
    return out


def _infer_signal_mask(
    volume: np.ndarray,
    *,
    threshold_rel: float = 0.01,
    background_frac_threshold: float = 0.05,
    abs_volume: np.ndarray | None = None,
    max_abs: float | None = None,
) -> np.ndarray:
    """
    Infer a "signal" mask for a volume.

    Heuristics:
    - If the volume is entirely zero, return an empty mask.
    - If the volume is constant non-zero (synthetic tests), treat the whole volume as signal.
    - If a single value dominates (often constant background after normalization),
      treat that value as background and exclude it.
    - Otherwise fall back to a relative-magnitude threshold.
    """
    abs_vol = np.asarray(abs_volume) if abs_volume is not None else np.abs(volume)
    if abs_vol.shape != volume.shape:
        raise ValueError(f"abs_volume shape must match volume: {abs_vol.shape} != {volume.shape}")
    max_abs = float(np.max(abs_vol)) if max_abs is None else float(max_abs)
    if max_abs <= 0.0:
        return np.zeros_like(volume, dtype=bool)

    # Constant non-zero volume: treat as all-signal.
    if float(np.std(volume)) < 1e-12:
        return np.ones_like(volume, dtype=bool)

    if volume.ndim != 3:
        raise ValueError(f"_infer_signal_mask expects a 3D volume, got shape={volume.shape}")

    # Estimate background as the median of corners (robust to a few non-background corners).
    z0, z1 = 0, volume.shape[0] - 1
    y0, y1 = 0, volume.shape[1] - 1
    x0, x1 = 0, volume.shape[2] - 1
    corners = np.array(
        [
            volume[z0, y0, x0],
            volume[z0, y0, x1],
            volume[z0, y1, x0],
            volume[z0, y1, x1],
            volume[z1, y0, x0],
            volume[z1, y0, x1],
            volume[z1, y1, x0],
            volume[z1, y1, x1],
        ],
        dtype=np.float64,
    )
    bg_val = float(np.median(corners))
    bg_mask = np.isclose(volume, bg_val, rtol=0.0, atol=1e-6)
    if float(bg_mask.mean()) >= float(background_frac_threshold):
        return ~bg_mask

    abs_threshold = float(threshold_rel) * max_abs
    return abs_vol > abs_threshold


def _coords_as_void(coords: np.ndarray) -> np.ndarray:
    coords = np.ascontiguousarray(coords)
    return coords.view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))).ravel()


def _dedupe_coords(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return coords
    view = _coords_as_void(coords)
    _, first_idx = np.unique(view, return_index=True)
    return coords[np.sort(first_idx)]


def sample_coords_sift_union_from_roi(
    extractor: Any,
    *,
    vol_a: np.ndarray,
    vol_b: np.ndarray,
    roi_mask: np.ndarray,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
    coords_pool: np.ndarray,
    n_goal: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Sample paired coordinates by unioning SIFT detections from both modalities.

    Strategy:
    - Detect ~n/2 keypoints in A and ~n/2 in B (within the same ROI/crop)
    - Union and deduplicate the coords
    - If fewer than n_goal coords are obtained, fill the remainder with random ROI samples

    Notes:
    - `extractor` must implement `detect_keypoints_3d(volume, roi_mask, crop_min, crop_max, n_keypoints)`.
    - Coordinates use the repo convention (z, y, x).
    """
    if not hasattr(extractor, "detect_keypoints_3d"):
        raise ValueError(
            "sift_union sampling requires extractor.detect_keypoints_3d(volume, roi_mask, crop_min, crop_max, n_keypoints)."
        )

    coords_pool = np.asarray(coords_pool, dtype=np.int32)
    if coords_pool.ndim != 2 or coords_pool.shape[1] != 3:
        raise ValueError(f"coords_pool must have shape (N, 3), got {coords_pool.shape}")
    if len(coords_pool) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    n_goal = int(n_goal)
    if n_goal <= 0:
        return np.zeros((0, 3), dtype=np.int32)
    n_goal = int(min(n_goal, len(coords_pool)))

    n_a = int((n_goal + 1) // 2)
    n_b = int(n_goal // 2)

    coords_a = extractor.detect_keypoints_3d(vol_a, roi_mask, crop_min, crop_max, n_a)
    coords_b = extractor.detect_keypoints_3d(vol_b, roi_mask, crop_min, crop_max, n_b)

    if coords_a.size == 0 and coords_b.size == 0:
        idx = rng.choice(len(coords_pool), size=n_goal, replace=False)
        return coords_pool[idx].astype(np.int32, copy=False)

    coords = np.vstack([c for c in (coords_a, coords_b) if c.size != 0]).astype(np.int32, copy=False)
    coords = _dedupe_coords(coords)
    if len(coords) >= n_goal:
        return coords[:n_goal].astype(np.int32, copy=False)

    # Fill remainder with random ROI samples not already selected.
    pool_view = _coords_as_void(coords_pool)
    selected_view = _coords_as_void(coords.astype(coords_pool.dtype, copy=False))
    remaining = coords_pool[~np.isin(pool_view, selected_view)]
    if len(remaining) == 0:
        return coords.astype(np.int32, copy=False)

    n_missing = int(min(n_goal - len(coords), len(remaining)))
    idx = rng.choice(len(remaining), size=n_missing, replace=False)
    return np.vstack([coords, remaining[idx]]).astype(np.int32, copy=False)


def sample_keypoints_2d_ref(
    extractor: Any,
    *,
    vol_ref: np.ndarray,
    roi_mask: np.ndarray,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
    n_goal: int,
    max_keypoints_per_slice: int = 500,
    rng: np.random.RandomState,  # noqa: ARG001 - reserved for future stochastic policies
    z_slices: list[int] | None = None,
) -> list[tuple[int, Any]]:
    """
    Sample 2D SIFT keypoints slice-by-slice from a reference volume.

    This returns keypoint geometry (x, y, size, angle) rather than voxel coordinates,
    enabling paired descriptor extraction by reusing the same keypoints across modalities.

    Notes:
    - `roi_mask` is expected to be a pre-eroded 3D boolean mask computed upstream.
    - Slices are processed in descending order of ROI area (within crop bounds).
    - Keypoints per slice are capped by top detector response.
    """
    if not hasattr(extractor, "detect_keypoints_2d"):
        raise ValueError("sample_keypoints_2d_ref requires extractor.detect_keypoints_2d(image, mask, max_keypoints).")

    n_goal = int(n_goal)
    if n_goal <= 0:
        return []

    max_keypoints_per_slice = int(max_keypoints_per_slice)
    if max_keypoints_per_slice <= 0:
        return []

    vol_ref = np.asarray(vol_ref)
    roi_mask = np.asarray(roi_mask).astype(bool, copy=False)

    if vol_ref.ndim != 3:
        raise ValueError(f"vol_ref must be 3D (D, H, W), got shape={vol_ref.shape}")
    if roi_mask.shape != vol_ref.shape:
        raise ValueError(f"roi_mask shape must match vol_ref: {roi_mask.shape} != {vol_ref.shape}")

    crop_min_arr = np.asarray(crop_min, dtype=np.int64)
    crop_max_arr = np.asarray(crop_max, dtype=np.int64)
    if crop_min_arr.ndim != 1 or crop_max_arr.ndim != 1:
        raise ValueError(
            "crop_min and crop_max must be 1D arrays with 3 elements (z, y, x); "
            f"got crop_min shape={crop_min_arr.shape}, crop_max shape={crop_max_arr.shape}"
        )
    if crop_min_arr.size != 3 or crop_max_arr.size != 3:
        raise ValueError(
            "crop_min and crop_max must have exactly 3 elements (z, y, x); "
            f"got crop_min shape={crop_min_arr.shape}, crop_max shape={crop_max_arr.shape}"
        )
    crop_min = crop_min_arr.reshape(3)
    crop_max = crop_max_arr.reshape(3)

    z0, z1 = int(crop_min[0]), int(crop_max[0])
    y0, y1 = int(crop_min[1]), int(crop_max[1])
    x0, x1 = int(crop_min[2]), int(crop_max[2])

    if not (0 <= z0 < z1 <= vol_ref.shape[0]):
        raise ValueError(f"Invalid crop z-bounds: [{z0}, {z1}) for vol_ref shape={vol_ref.shape}")
    if not (0 <= y0 < y1 <= vol_ref.shape[1]):
        raise ValueError(f"Invalid crop y-bounds: [{y0}, {y1}) for vol_ref shape={vol_ref.shape}")
    if not (0 <= x0 < x1 <= vol_ref.shape[2]):
        raise ValueError(f"Invalid crop x-bounds: [{x0}, {x1}) for vol_ref shape={vol_ref.shape}")

    # Determine slice order: use provided z_slices or rank by ROI area.
    if z_slices is not None:
        # Use provided slices (filter to valid range with ROI)
        slices_to_process = []
        for z in z_slices:
            z = int(z)
            if z0 <= z < z1:
                area = int(roi_mask[z, y0:y1, x0:x1].sum())
                if area > 0:
                    slices_to_process.append(z)
    else:
        # Rank slices by ROI area within crop bounds (default behavior).
        slice_scores: list[tuple[int, int]] = []
        for z in range(z0, z1):
            area = int(roi_mask[z, y0:y1, x0:x1].sum())
            if area > 0:
                slice_scores.append((area, z))

        if not slice_scores:
            return []

        slice_scores.sort(key=lambda t: (-t[0], t[1]))
        slices_to_process = [z for _, z in slice_scores]

    if not slices_to_process:
        return []

    selected: list[tuple[int, Any]] = []

    # When z_slices is explicitly provided, process ALL specified slices
    # (don't stop early at n_goal). This ensures fair comparison when using
    # --num_slices to request a specific number of slices.
    # When z_slices is None (SIFT-based selection), stop early at n_goal for efficiency.
    force_all_slices = z_slices is not None

    for z in slices_to_process:
        if not force_all_slices and len(selected) >= n_goal:
            break

        roi = roi_mask[z, y0:y1, x0:x1]
        if not bool(np.any(roi)):
            continue

        mask_u8 = np.zeros_like(roi_mask[z], dtype=np.uint8)
        mask_u8[y0:y1, x0:x1] = roi.astype(np.uint8) * 255

        # When forcing all slices, cap keypoints per slice proportionally
        # to distribute n_goal across all slices
        if force_all_slices:
            # Use n_goal / num_slices as approximate per-slice budget, with minimum of 1
            per_slice_budget = max(1, n_goal // len(slices_to_process))
            effective_max_kpts = min(max_keypoints_per_slice, per_slice_budget)
        else:
            effective_max_kpts = max_keypoints_per_slice

        kps = extractor.detect_keypoints_2d(
            vol_ref[z],
            mask=mask_u8,
            max_keypoints=effective_max_kpts,
        )
        if not kps:
            continue

        if force_all_slices:
            # Take all detected keypoints (already capped by effective_max_kpts)
            selected.extend((int(z), kp) for kp in kps)
        else:
            n_take = int(min(len(kps), n_goal - len(selected)))
            selected.extend((int(z), kp) for kp in kps[:n_take])

    return selected


def sample_keypoints_2d_dual(
    extractor: Any,
    *,
    vol_a: np.ndarray,
    vol_b: np.ndarray,
    roi_mask: np.ndarray,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
    n_goal: int,
    max_keypoints_per_slice: int = 500,
    rng: np.random.RandomState,
    z_slices: list[int] | None = None,
) -> tuple[list[tuple[int, Any]], list[tuple[int, Any]]]:
    """
    Sample keypoints in both modalities, returning two anchor sets.

    Returns:
        (kps_a_anchored, kps_b_anchored) where each list contains (z, cv2.KeyPoint).
    """
    n_goal = int(n_goal)
    if n_goal <= 0:
        return [], []

    n_a = int((n_goal + 1) // 2)
    n_b = int(n_goal // 2)

    kps_a = sample_keypoints_2d_ref(
        extractor,
        vol_ref=vol_a,
        roi_mask=roi_mask,
        crop_min=crop_min,
        crop_max=crop_max,
        n_goal=n_a,
        max_keypoints_per_slice=max_keypoints_per_slice,
        rng=rng,
        z_slices=z_slices,
    )
    kps_b = sample_keypoints_2d_ref(
        extractor,
        vol_ref=vol_b,
        roi_mask=roi_mask,
        crop_min=crop_min,
        crop_max=crop_max,
        n_goal=n_b,
        max_keypoints_per_slice=max_keypoints_per_slice,
        rng=rng,
        z_slices=z_slices,
    )
    return kps_a, kps_b


def select_slices_uniform_random(
    *,
    roi_mask: np.ndarray,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
    n_slices: int,
    rng: np.random.RandomState,
    min_roi_area: int = 100,
) -> list[int]:
    """
    Select slices uniformly at random from valid ROI slices.

    This provides a fairer comparison between methods by using the same
    randomly selected slices rather than method-specific slice selection.

    Args:
        roi_mask: 3D boolean mask indicating valid ROI voxels
        crop_min: (3,) array with minimum crop bounds (z, y, x)
        crop_max: (3,) array with maximum crop bounds (z, y, x)
        n_slices: Number of slices to select
        rng: Random state for reproducibility
        min_roi_area: Minimum ROI area (in voxels) required for a slice to be valid

    Returns:
        List of z-indices of selected slices (sorted ascending)
    """
    roi_mask = np.asarray(roi_mask).astype(bool, copy=False)
    if roi_mask.ndim != 3:
        raise ValueError(f"roi_mask must be 3D, got shape={roi_mask.shape}")

    crop_min_arr = np.asarray(crop_min, dtype=np.int64).reshape(3)
    crop_max_arr = np.asarray(crop_max, dtype=np.int64).reshape(3)

    z0, z1 = int(crop_min_arr[0]), int(crop_max_arr[0])
    y0, y1 = int(crop_min_arr[1]), int(crop_max_arr[1])
    x0, x1 = int(crop_min_arr[2]), int(crop_max_arr[2])

    # Find valid slices with sufficient ROI area
    valid_slices: list[int] = []
    for z in range(z0, z1):
        area = int(roi_mask[z, y0:y1, x0:x1].sum())
        if area >= min_roi_area:
            valid_slices.append(z)

    if not valid_slices:
        return []

    n_slices = int(min(n_slices, len(valid_slices)))
    if n_slices <= 0:
        return []

    # Uniform random selection
    selected_idx = rng.choice(len(valid_slices), size=n_slices, replace=False)
    selected = sorted([valid_slices[i] for i in selected_idx])

    return selected


def select_slices_by_roi_area(
    *,
    roi_mask: np.ndarray,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
    n_slices: int,
    min_roi_area: int = 100,
) -> list[int]:
    """Select the top-N slices ranked by ROI area (largest first).

    This is the deterministic counterpart of :func:`select_slices_uniform_random`.
    When ``n_slices`` is large enough to include all valid slices, the result is
    identical to the default SIFT-based evaluation (which uses all valid slices).

    Args:
        roi_mask: 3D boolean mask indicating valid ROI voxels.
        crop_min: (3,) array with minimum crop bounds (z, y, x).
        crop_max: (3,) array with maximum crop bounds (z, y, x).
        n_slices: Maximum number of slices to select.
        min_roi_area: Minimum ROI area (in voxels) for a slice to be valid.

    Returns:
        List of z-indices of selected slices (sorted ascending).
    """
    roi_mask = np.asarray(roi_mask).astype(bool, copy=False)
    if roi_mask.ndim != 3:
        raise ValueError(f"roi_mask must be 3D, got shape={roi_mask.shape}")

    crop_min_arr = np.asarray(crop_min, dtype=np.int64).reshape(3)
    crop_max_arr = np.asarray(crop_max, dtype=np.int64).reshape(3)

    z0, z1 = int(crop_min_arr[0]), int(crop_max_arr[0])
    y0, y1 = int(crop_min_arr[1]), int(crop_max_arr[1])
    x0, x1 = int(crop_min_arr[2]), int(crop_max_arr[2])

    # Collect (area, z) for each valid slice
    slice_areas: list[tuple[int, int]] = []
    for z in range(z0, z1):
        area = int(roi_mask[z, y0:y1, x0:x1].sum())
        if area >= min_roi_area:
            slice_areas.append((area, z))

    if not slice_areas:
        return []

    # Sort by area descending, take top n_slices
    slice_areas.sort(key=lambda t: -t[0])
    n_slices = min(n_slices, len(slice_areas))
    selected = sorted([z for _, z in slice_areas[:n_slices]])

    return selected
