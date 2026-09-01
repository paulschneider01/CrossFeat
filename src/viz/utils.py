"""Shared visualization utilities for CrossFeat."""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import affine_transform

# Color scheme for visualizations
COLORS = {
    "inlier": "#00FF00",  # Green
    "outlier": "#FF4444",  # Red
    "crossing_enabled": "#2ecc71",  # Emerald green
    "crossing_disabled": "#e74c3c",  # Alizarin red
    "tta_enabled": "#3498db",  # Peter river blue
    "tta_disabled": "#95a5a6",  # Concrete gray
}

# Default figure settings
FIGURE_DPI = 150
FIGURE_FACECOLOR = "white"


def normalize_for_display(volume: np.ndarray, percentile: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """
    Normalize volume for display by clipping to percentile range.

    Args:
        volume: Input volume
        percentile: (low, high) percentiles for clipping

    Returns:
        Normalized volume in [0, 1] range
    """
    low = np.percentile(volume, percentile[0])
    high = np.percentile(volume, percentile[1])

    if high <= low:
        return np.zeros_like(volume)

    normalized = (volume - low) / (high - low)
    return np.clip(normalized, 0, 1)


def get_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """
    Extract a 2D slice from a 3D volume.

    Args:
        volume: 3D volume
        axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial)
        index: Slice index

    Returns:
        2D slice
    """
    if axis == 0:
        return volume[index, :, :]
    elif axis == 1:
        return volume[:, index, :]
    else:  # axis == 2
        return volume[:, :, index]


def get_slice_coords(coords: np.ndarray, axis: int) -> Tuple[np.ndarray, int]:
    """
    Get 2D coordinates for a given slice axis.

    Args:
        coords: Nx3 coordinates
        axis: Axis to project onto

    Returns:
        (2D coordinates, slice axis index)
    """
    if axis == 0:
        return coords[:, [1, 2]], 0  # (y, z)
    elif axis == 1:
        return coords[:, [0, 2]], 1  # (x, z)
    else:  # axis == 2
        return coords[:, [0, 1]], 2  # (x, y)


def create_colormap_by_tre(tre_values: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    Create RGBA colors based on TRE values.

    Green for inliers (TRE < threshold), red for outliers.

    Args:
        tre_values: Target registration errors
        threshold: Inlier threshold

    Returns:
        Nx4 RGBA array
    """
    is_inlier = tre_values < threshold
    colors = np.zeros((len(tre_values), 4))

    # Inliers: green
    colors[is_inlier] = mcolors.to_rgba(COLORS["inlier"])

    # Outliers: red with alpha based on TRE
    outlier_mask = ~is_inlier
    colors[outlier_mask] = mcolors.to_rgba(COLORS["outlier"])

    # Reduce alpha for large outliers
    if np.any(outlier_mask):
        outlier_tre = tre_values[outlier_mask]
        max_tre = np.percentile(outlier_tre, 95)
        alpha_scale = np.clip(outlier_tre / max_tre, 0, 1)
        colors[outlier_mask, 3] = 0.3 + 0.5 * (1 - alpha_scale)

    return colors


def setup_figure_style():
    """Configure matplotlib style for consistent visualizations."""
    plt.rcParams.update({
        "figure.facecolor": FIGURE_FACECOLOR,
        "axes.facecolor": FIGURE_FACECOLOR,
        "savefig.facecolor": FIGURE_FACECOLOR,
        "savefig.dpi": FIGURE_DPI,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


def format_metrics_text(
    num_matches: int,
    num_inliers: int,
    inlier_ratio: float,
    tre_mean: float,
    tre_median: Optional[float] = None,
) -> str:
    """
    Format metrics as a display string.

    Args:
        num_matches: Number of mutual NN matches
        num_inliers: Number of inliers
        inlier_ratio: Inlier ratio (0-1)
        tre_mean: Mean TRE
        tre_median: Median TRE (optional)

    Returns:
        Formatted string
    """
    lines = [
        f"Matches: {num_matches}",
        f"Inliers: {num_inliers} ({inlier_ratio:.1%})",
        f"TRE: {tre_mean:.2f}" + (f" (med: {tre_median:.2f})" if tre_median is not None else ""),
    ]
    return "\n".join(lines)


def get_transformed_axis_mapping(rot_mat: np.ndarray) -> dict:
    """
    Determine how anatomical axes map after rotation.

    Given a rotation matrix R, find which array axis in the transformed volume
    corresponds to each original anatomical axis. This is needed because after
    rotating the volume content, slicing along axis=2 no longer gives an axial
    anatomical view - it gives whatever plane the original Z axis rotated to.

    Args:
        rot_mat: 3x3 rotation matrix

    Returns:
        Dictionary mapping original axis (0, 1, 2) to transformed axis (0, 1, 2)
        e.g., {0: 2, 1: 1, 2: 0} means original X maps to transformed Z, etc.
    """
    # Standard basis vectors for each axis
    # axis 0 (X/Sagittal), axis 1 (Y/Coronal), axis 2 (Z/Axial)
    basis = np.eye(3)

    axis_mapping = {}
    for orig_axis in range(3):
        # The original axis direction
        orig_direction = basis[orig_axis]

        # After rotation, this direction becomes R @ orig_direction
        transformed_direction = rot_mat @ orig_direction

        # Find which axis the transformed direction is closest to
        # (taking absolute value since we don't care about sign/direction)
        closest_axis = np.argmax(np.abs(transformed_direction))
        axis_mapping[orig_axis] = closest_axis

    return axis_mapping


def transform_volume_for_display(
    volume: np.ndarray,
    rot_mat: np.ndarray,
    trans_vec: np.ndarray,
) -> np.ndarray:
    """
    Apply a linear transform and translation to a volume for visualization.

    This simulates the "in the wild" misalignment by transforming vol_b
    so it appears misaligned relative to vol_a.

    Args:
        volume: 3D volume to transform
        rot_mat: 3x3 linear transform matrix
        trans_vec: 3-element translation vector

    Returns:
        Transformed volume
    """
    # Center of the volume
    center = (np.array(volume.shape, dtype=np.float64) - 1.0) / 2.0

    # Build the affine transformation matrix for scipy.ndimage.affine_transform
    # The transform maps output coords -> input coords (inverse mapping)
    # Our forward transform is: out = A @ (in - center) + center + trans
    # Inverse: in = A^{-1} @ (out - center - trans) + center
    #            = A^{-1} @ out + (center - A^{-1} @ center - A^{-1} @ trans)

    rot_mat = np.asarray(rot_mat, dtype=np.float64).reshape(3, 3)
    trans_vec = np.asarray(trans_vec, dtype=np.float64).reshape(3)
    rot_inv = np.linalg.inv(rot_mat)

    # Offset for the inverse transform
    offset = center - rot_inv @ center - rot_inv @ trans_vec

    # Apply the affine transform
    transformed = affine_transform(
        volume,
        rot_inv,
        offset=offset,
        order=1,  # Bilinear interpolation
        mode='constant',
        cval=0.0,
    )

    return transformed


def _plane_axes_for_slice_axis(axis: int) -> Tuple[int, int]:
    """
    Return the two coordinate axes that span the slice plane.

    Conventions match `get_slice` / `get_slice_coords`:
      - axis=0 (sagittal): plane is (y, z)
      - axis=1 (coronal):  plane is (x, z)
      - axis=2 (axial):    plane is (x, y)
    """
    if axis == 0:
        return 1, 2
    if axis == 1:
        return 0, 2
    return 0, 1


def _nearest_rotation_matrix_2d(mat: np.ndarray) -> np.ndarray:
    """Project a 2x2 matrix to the nearest proper 2D rotation matrix."""
    u, _, vt = np.linalg.svd(mat)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def _inplane_transform_2d(
    rot_mat: np.ndarray,
    trans_vec: np.ndarray,
    center_3d: np.ndarray,
    slice_axis: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute an in-plane 2D transform for a given slice axis.

    This intentionally ignores out-of-plane components of the 3D transform so that we
    can visualize misalignment consistently in 2D (same slice index).

    Behavior:
    - If `rot_mat` is (approximately) a proper 3D rotation matrix, we project the
      in-plane 2x2 block to the nearest proper 2D rotation (preserving prior behavior).
    - Otherwise, we treat `rot_mat` as a general invertible linear transform and keep
      the raw in-plane 2x2 block (enabling affine visualization, e.g. scale/shear).
    """
    ax0, ax1 = _plane_axes_for_slice_axis(slice_axis)
    rot_mat = np.asarray(rot_mat, dtype=np.float64).reshape(3, 3)
    rot_2d_raw = rot_mat[np.ix_([ax0, ax1], [ax0, ax1])]

    # Detect whether the 3D matrix is a (near) rotation; if so, keep legacy projection.
    rt_r = rot_mat.T @ rot_mat
    is_rotation = np.allclose(rt_r, np.eye(3), atol=1e-3) and np.isclose(np.linalg.det(rot_mat), 1.0, atol=1e-3)
    rot_2d = _nearest_rotation_matrix_2d(rot_2d_raw) if is_rotation else rot_2d_raw
    trans_2d = trans_vec[[ax0, ax1]]
    center_2d = center_3d[[ax0, ax1]]
    return rot_2d, trans_2d, center_2d


def transform_slice_for_display(
    slice_2d: np.ndarray,
    rot_mat: np.ndarray,
    trans_vec: np.ndarray,
    center_3d: np.ndarray,
    slice_axis: int,
) -> np.ndarray:
    """
    Apply an in-plane 2D transform to a 2D slice for visualization.

    The output slice stays in the same array grid (same shape); content is rotated
    and translated within the plane.
    """
    rot_2d, trans_2d, center_2d = _inplane_transform_2d(rot_mat, trans_vec, center_3d, slice_axis)

    rot_inv = np.linalg.inv(rot_2d)
    offset = center_2d - rot_inv @ center_2d - rot_inv @ trans_2d

    return affine_transform(
        slice_2d,
        rot_inv,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
    )


def transform_coords_for_display(
    coords_2d: np.ndarray,
    rot_mat: np.ndarray,
    trans_vec: np.ndarray,
    center_3d: np.ndarray,
    slice_axis: int,
) -> np.ndarray:
    """Apply the same in-plane 2D transform (used for `transform_slice_for_display`) to 2D points."""
    if coords_2d.size == 0:
        return coords_2d

    rot_2d, trans_2d, center_2d = _inplane_transform_2d(rot_mat, trans_vec, center_3d, slice_axis)
    centered = coords_2d - center_2d
    return (rot_2d @ centered.T).T + center_2d + trans_2d
