"""2D-safe ROI preprocessing utilities.

Provides margin/erosion functions that adapt per-axis to handle depth=1 volumes
(single-slice 2D data) safely, avoiding empty masks and impossible coordinate ranges.

The invariant: margin_i = min(base_margin, (dim_i - 1) // 2) guarantees non-empty
bounds for ANY positive dimension.
"""

import numpy as np
from scipy.ndimage import binary_erosion


def compute_safe_margin(dim: int, base_margin: int = 15) -> int:
    """Compute per-axis margin: min(base_margin, (dim - 1) // 2).

    Guarantees non-empty bounds for ANY dimension:
    - dim=1 -> margin=0, range [0, 1)
    - dim=20 -> margin=9, range [9, 11)
    - dim=128 -> margin=15, range [15, 113)

    Args:
        dim: Size of the axis.
        base_margin: Desired margin (default 15).

    Returns:
        Safe margin value for this axis.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if base_margin < 0:
        raise ValueError(f"base_margin must be non-negative, got {base_margin}")
    return min(base_margin, (dim - 1) // 2)


def safe_preprocess_roi(
    mask: np.ndarray,
    erode_iterations: int = 5,
    base_margin: int = 15,
) -> np.ndarray:
    """Apply erosion and margin, auto-adapting per axis.

    Structuring element per axis configuration:
    - depth=1: 2D structuring element np.ones((1,3,3)) -- no erosion along z
    - depth>1 but compute_safe_margin(depth)==0: same (1,3,3) element
    - depth>1 with margin>0: standard 3D cross element (scipy default)

    Args:
        mask: 3D boolean mask (D, H, W).
        erode_iterations: Number of erosion iterations.
        base_margin: Base margin for boundary exclusion.

    Returns:
        Preprocessed boolean mask with erosion applied and margins enforced.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D (D, H, W), got shape={mask.shape}")

    if erode_iterations <= 0:
        eroded = mask.copy()
    elif mask.mean() < 0.3:
        # Skip erosion for sparse masks (< 30% coverage). Projected
        # sensor data (LiDAR ~2%, event camera ~4-11%) has scattered
        # valid pixels that don't survive morphological erosion.
        eroded = mask.copy()
    else:
        d, h, w = mask.shape
        z_margin = compute_safe_margin(d, base_margin)

        if d == 1 or z_margin == 0:
            # 2D structuring element: no erosion along z
            struct = np.ones((1, 3, 3), dtype=bool)
        else:
            # Standard 3D cross element (scipy default)
            struct = None

        eroded = binary_erosion(mask, structure=struct, iterations=erode_iterations)
        # Ensure boolean array (scipy may return ndarray)
        eroded = np.asarray(eroded, dtype=bool)

    # Apply per-axis margins
    d, h, w = eroded.shape
    mz = compute_safe_margin(d, base_margin)
    my = compute_safe_margin(h, base_margin)
    mx = compute_safe_margin(w, base_margin)

    # Zero out regions outside margins
    if mz > 0:
        eroded[:mz, :, :] = False
        eroded[d - mz :, :, :] = False
    if my > 0:
        eroded[:, :my, :] = False
        eroded[:, h - my :, :] = False
    if mx > 0:
        eroded[:, :, :mx] = False
        eroded[:, :, w - mx :] = False

    return eroded


def compute_safe_crop_bounds(
    shape: tuple[int, ...],
    base_margin: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (crop_min, crop_max) using compute_safe_margin per axis.

    margin_i = compute_safe_margin(shape[i], base_margin)
    crop_min[i] = margin_i
    crop_max[i] = shape[i] - margin_i
    Always guarantees crop_min[i] < crop_max[i] for any positive dimension.

    Args:
        shape: Volume shape tuple (e.g., (D, H, W)).
        base_margin: Base margin for boundary exclusion.

    Returns:
        (crop_min, crop_max) as int32 numpy arrays.
    """
    margins = np.array(
        [compute_safe_margin(int(s), base_margin) for s in shape], dtype=np.int32
    )
    shape_arr = np.array(shape, dtype=np.int32)
    crop_min = margins
    crop_max = shape_arr - margins
    return crop_min, crop_max
