"""
Visualization functions for CrossFeat evaluation.

Provides functions to create visual sanity checks for cross-modal matching:
- Multi-view match overlays (axial, coronal, sagittal)
- TRE distribution histograms
- Crossing comparison figures
- TTA comparison figures
- Summary figures across cases
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import numpy as np

from src.viz.utils import (
    COLORS,
    FIGURE_DPI,
    FIGURE_FACECOLOR,
    normalize_for_display,
    get_slice,
    get_slice_coords,
    format_metrics_text,
    transform_slice_for_display,
    transform_coords_for_display,
)


@dataclass
class VisualizationData:
    """Data needed for visualization of a single case."""
    case_id: str
    vol_a: np.ndarray  # Source modality volume
    vol_b: np.ndarray  # Target modality volume
    coords_a: np.ndarray  # Sampled coordinates in A
    coords_b: np.ndarray  # Corresponding coordinates in B (may differ if misaligned)
    match_idx_a: np.ndarray  # Mutual NN match indices into coords_a
    match_idx_b: np.ndarray  # Mutual NN match indices into coords_b
    tre_values: np.ndarray  # Per-match TRE
    is_inlier: np.ndarray  # Boolean inlier mask for matches
    d_a_crossed: np.ndarray  # Crossed descriptors
    d_b_norm: np.ndarray  # Target descriptors (normalized)
    modality_a: str = "A"
    modality_b: str = "B"
    # Metrics
    top1_acc: float = 0.0
    inlier_ratio: float = 0.0
    tre_mean: float = 0.0
    tre_median: float = 0.0
    # Transformation info (for "in the wild" visualization)
    rotate_deg: float = 0.0
    translate_mm: float = 0.0
    affine_scale: float = 0.0
    affine_shear_deg: float = 0.0
    # Actual transformation matrices (for applying to volume display)
    rot_mat: Optional[np.ndarray] = None  # 3x3 linear transform matrix
    trans_vec: Optional[np.ndarray] = None  # 3-element translation vector
    # Whether evaluation used strictly in-plane (2D) augmentation.
    two_d_only: bool = True

    @property
    def is_transformed(self) -> bool:
        """Whether transformation (misalignment) was applied."""
        return (
            self.rotate_deg > 0
            or self.translate_mm > 0
            or self.affine_scale > 0
            or self.affine_shear_deg > 0
        )

    @property
    def transform_str(self) -> str:
        """Human-readable transformation description."""
        if not self.is_transformed:
            return ""
        parts = []
        if self.rotate_deg > 0:
            parts.append(f"rot={self.rotate_deg:.0f}deg")
        if self.translate_mm > 0:
            parts.append(f"trans={self.translate_mm:.0f}mm")
        if self.affine_scale > 0:
            parts.append(f"scale±{self.affine_scale:.2f}")
        if self.affine_shear_deg > 0:
            parts.append(f"shear±{self.affine_shear_deg:.1f}°")
        return " (" + ", ".join(parts) + ")"


def _get_display_volumes(viz_data: VisualizationData) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare volumes for display (misalignment is applied slice-level for 2D visualization)."""
    return normalize_for_display(viz_data.vol_a), normalize_for_display(viz_data.vol_b)


def _plot_slice_with_matches(
    ax1: plt.Axes,
    ax2: plt.Axes,
    fig: plt.Figure,
    slice_a: np.ndarray,
    slice_b: np.ndarray,
    coords_a_2d: np.ndarray,
    coords_b_2d: np.ndarray,
    is_inlier: np.ndarray,
    show_outliers: bool = True,
    title_a: str = "Source",
    title_b: str = "Target",
) -> Tuple[int, int]:
    """
    Plot a pair of slices with matched keypoints and connecting lines.

    Returns:
        (num_inliers_shown, num_outliers_shown)
    """
    # Display slices
    ax1.imshow(slice_a.T, cmap='gray', origin='lower', aspect='equal')
    ax2.imshow(slice_b.T, cmap='gray', origin='lower', aspect='equal')

    n_inliers = 0
    n_outliers = 0

    for i in range(len(coords_a_2d)):
        inlier = is_inlier[i]

        if inlier:
            color = COLORS["inlier"]
            alpha = 0.9
            linewidth = 1.5
            markersize = 8
            n_inliers += 1
        else:
            if not show_outliers:
                continue
            color = COLORS["outlier"]
            alpha = 0.4
            linewidth = 0.8
            markersize = 5
            n_outliers += 1

        # Plot keypoints
        ax1.scatter(
            coords_a_2d[i, 0], coords_a_2d[i, 1],
            c=color, s=markersize**2, marker='o',
            edgecolors='white', linewidths=0.5, alpha=alpha
        )
        ax2.scatter(
            coords_b_2d[i, 0], coords_b_2d[i, 1],
            c=color, s=markersize**2, marker='o',
            edgecolors='white', linewidths=0.5, alpha=alpha
        )

        # Draw connecting line
        con = ConnectionPatch(
            xyA=(coords_a_2d[i, 0], coords_a_2d[i, 1]),
            xyB=(coords_b_2d[i, 0], coords_b_2d[i, 1]),
            coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2,
            color=color, alpha=alpha * 0.6, linewidth=linewidth
        )
        fig.add_artist(con)

    ax1.set_title(f"{title_a}\n({n_inliers} inliers, {n_outliers} outliers)")
    ax2.set_title(title_b)
    ax1.axis('off')
    ax2.axis('off')

    return n_inliers, n_outliers


def plot_matches_multiview(
    viz_data: VisualizationData,
    output_path: Path,
    window: int = 15,
    show_outliers: bool = True,
    fixed_slice_z: Optional[int] = None,
) -> Optional[int]:
    """
    Create a multi-view visualization showing matches in 3 orthogonal views.

    Args:
        viz_data: Visualization data for a single case
        output_path: Path to save the figure
        window: Include matches within this distance from slice
        show_outliers: Whether to show outlier matches
        fixed_slice_z: If provided, use this z-slice instead of auto-selecting.
            Useful for comparing different methods on the same slice.

    Returns:
        The z-slice index used for the axial view (useful for synchronizing
        visualizations across methods).
    """
    # Volume convention across the codebase is (z, y, x) -> axes (0, 1, 2).
    # In 2D-only mode we restrict visualization to axial slices (z fixed).
    axes_order = [0, 1, 2]
    view_names = ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]
    if viz_data.two_d_only:
        axes_order = [0]
        view_names = ["Axial (Z)"]

    n_views = len(axes_order)
    # Extra height for title and legend
    fig = plt.figure(figsize=(18, 5 * n_views + 1.5))

    # Get matched coordinates (handle empty case)
    if len(viz_data.match_idx_a) > 0:
        matched_coords_a = viz_data.coords_a[viz_data.match_idx_a]
        matched_coords_b = viz_data.coords_b[viz_data.match_idx_b]
    else:
        matched_coords_a = np.array([]).reshape(0, 3)
        matched_coords_b = np.array([]).reshape(0, 3)

    # Find best slices separately for each volume (where most inliers are)
    # This is crucial for misaligned volumes where coords_a and coords_b differ
    if len(matched_coords_a) > 0 and len(viz_data.is_inlier) == len(matched_coords_a):
        inlier_coords_a = matched_coords_a[viz_data.is_inlier]
        inlier_coords_b = matched_coords_b[viz_data.is_inlier]
    else:
        inlier_coords_a = np.array([]).reshape(0, 3)
        inlier_coords_b = np.array([]).reshape(0, 3)

    if fixed_slice_z is not None:
        # Use the provided fixed slice for z-axis (axial view)
        best_slices_a = np.array([fixed_slice_z, viz_data.vol_a.shape[1] // 2, viz_data.vol_a.shape[2] // 2])
        best_slices_b = np.array([fixed_slice_z, viz_data.vol_b.shape[1] // 2, viz_data.vol_b.shape[2] // 2])
    elif len(inlier_coords_a) > 0:
        best_slices_a = np.round(np.median(inlier_coords_a, axis=0)).astype(int)
        best_slices_b = np.round(np.median(inlier_coords_b, axis=0)).astype(int)
    else:
        best_slices_a = np.array(viz_data.vol_a.shape) // 2
        best_slices_b = np.array(viz_data.vol_b.shape) // 2

    # Title
    num_matches = len(viz_data.match_idx_a)
    num_inliers = int(np.sum(viz_data.is_inlier)) if len(viz_data.is_inlier) > 0 else 0
    title_color = COLORS["crossing_enabled"]

    # Add transformation info to title if applicable
    transform_info = viz_data.transform_str if viz_data.is_transformed else ""
    misalign_label = " [MISALIGNED]" if viz_data.is_transformed else ""

    fig.suptitle(
        f"{viz_data.case_id}: {viz_data.modality_a.upper()} <-> {viz_data.modality_b.upper()}{misalign_label}\n"
        f"Matches: {num_matches} | Inliers: {num_inliers} ({viz_data.inlier_ratio:.1%}) | "
        f"TRE: {viz_data.tre_mean:.2f} vox | Top-1: {viz_data.top1_acc:.1%}{transform_info}",
        fontsize=14, fontweight='bold', color=title_color, y=0.98
    )

    # Use top margin to prevent suptitle from overlapping subplot titles
    gs = gridspec.GridSpec(n_views, 2, figure=fig, hspace=0.3, wspace=0.1, top=0.82, bottom=0.10)

    # Normalize volumes for display
    vol_a_disp, vol_b_disp = _get_display_volumes(viz_data)
    center_b = (np.array(viz_data.vol_b.shape, dtype=np.float64) - 1.0) / 2.0

    for row, (axis, view_name) in enumerate(zip(axes_order, view_names)):
        # Slice both volumes at the same axis for consistent anatomical views
        slice_idx_a = int(np.clip(best_slices_a[axis], 0, viz_data.vol_a.shape[axis] - 1))
        slice_idx_b = int(np.clip(best_slices_a[axis], 0, viz_data.vol_b.shape[axis] - 1))

        ax1 = fig.add_subplot(gs[row, 0])
        ax2 = fig.add_subplot(gs[row, 1])

        # Get slices from same anatomical position in both volumes
        slice_a = get_slice(vol_a_disp, axis, slice_idx_a)
        slice_b = get_slice(vol_b_disp, axis, slice_idx_b)
        if viz_data.is_transformed and viz_data.rot_mat is not None and viz_data.trans_vec is not None:
            slice_b = transform_slice_for_display(slice_b, viz_data.rot_mat, viz_data.trans_vec, center_b, axis)

        # Get 2D coordinates (handle empty case)
        # For vol_a: coords are in vol_a space
        # For vol_b: coords are already in the evaluation (possibly transformed) space.
        if len(matched_coords_a) > 0:
            coords_a_2d, _ = get_slice_coords(matched_coords_a, axis)
            coords_b_2d, _ = get_slice_coords(matched_coords_b, axis)

            # Filter matches near this anatomical slice
            near_slice_a = np.abs(matched_coords_a[:, axis] - slice_idx_a) < window
            near_slice = near_slice_a  # Filter based on vol_a slice

            coords_a_2d_filtered = coords_a_2d[near_slice]
            coords_b_2d_filtered = coords_b_2d[near_slice]
            # Handle mismatch between is_inlier and match indices
            if len(viz_data.is_inlier) == len(matched_coords_a):
                is_inlier_filtered = viz_data.is_inlier[near_slice]
            else:
                is_inlier_filtered = np.zeros(near_slice.sum(), dtype=bool)
        else:
            coords_a_2d_filtered = np.array([]).reshape(0, 2)
            coords_b_2d_filtered = np.array([]).reshape(0, 2)
            is_inlier_filtered = np.array([], dtype=bool)

        # Labels show the slice index and misalignment indicator
        misalign_suffix = " [MISALIGNED]" if viz_data.is_transformed else ""
        _plot_slice_with_matches(
            ax1, ax2, fig,
            slice_a, slice_b,
            coords_a_2d_filtered, coords_b_2d_filtered,
            is_inlier_filtered,
            show_outliers=show_outliers,
            title_a=f"{viz_data.modality_a.upper()} - {view_name} (slice {slice_idx_a})",
            title_b=f"{viz_data.modality_b.upper()} - {view_name} (slice {slice_idx_b}){misalign_suffix}",
        )

    # Add legend at fixed position below plots
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["inlier"],
                   markersize=10, label='Inlier (correct match)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["outlier"],
                   markersize=10, label='Outlier (incorrect match)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 0.01))

    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.close()

    # Return the z-slice used for axial view (axis 0)
    return int(best_slices_a[0])


def plot_tre_distribution(
    viz_data: VisualizationData,
    output_path: Path,
    threshold: float = 5.0,
) -> None:
    """
    Plot the distribution of Target Registration Error.

    Args:
        viz_data: Visualization data
        output_path: Path to save the figure
        threshold: Inlier threshold in voxels
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    tre = viz_data.tre_values
    is_inlier = viz_data.is_inlier

    if tre is None or len(tre) == 0:
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{viz_data.case_id}: TRE Analysis (no matches)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=FIGURE_FACECOLOR)
        plt.close()
        return

    # TRE histogram
    ax = axes[0]
    inlier_tre = tre[is_inlier]
    outlier_tre = tre[~is_inlier]

    if len(inlier_tre) > 0:
        n_bins = min(30, max(1, len(inlier_tre) // 2))
        ax.hist(inlier_tre, bins=n_bins, alpha=0.7, color='green',
                label=f'Inliers (n={len(inlier_tre)})', edgecolor='darkgreen')
    if len(outlier_tre) > 0:
        n_bins = min(30, max(1, len(outlier_tre) // 2))
        ax.hist(outlier_tre, bins=n_bins, alpha=0.5, color='red',
                label=f'Outliers (n={len(outlier_tre)})', edgecolor='darkred')

    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold} vox)')
    ax.set_xlabel('TRE (voxels)')
    ax.set_ylabel('Count')
    ax.set_title('Target Registration Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative TRE
    ax = axes[1]
    sorted_tre = np.sort(tre)
    cumulative = np.arange(1, len(sorted_tre) + 1) / len(sorted_tre)

    ax.plot(sorted_tre, cumulative * 100, 'b-', linewidth=2)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold} vox)')
    ax.axhline(viz_data.inlier_ratio * 100, color='green', linestyle=':', linewidth=2,
               label=f'Inlier ratio ({viz_data.inlier_ratio:.1%})')
    ax.set_xlabel('TRE (voxels)')
    ax.set_ylabel('Cumulative %')
    ax.set_title('Cumulative TRE Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    x_max = float(sorted_tre.max()) if len(sorted_tre) > 0 else 0.0
    ax.set_xlim(0, min(50.0, max(threshold * 1.1, x_max * 1.1)))
    ax.set_ylim(0, 100)

    fig.suptitle(
        f"{viz_data.case_id}: TRE Analysis | Mean: {viz_data.tre_mean:.2f} | Median: {viz_data.tre_median:.2f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.close()


def plot_crossing_comparison(
    viz_no_cross: VisualizationData,
    viz_cross: VisualizationData,
    output_path: Path,
    slice_axis: int = 0,
) -> None:
    """
    Create a side-by-side comparison of matching with and without crossing.

    Args:
        viz_no_cross: Visualization data without crossing
        viz_cross: Visualization data with crossing
        output_path: Path to save the figure
        slice_axis: Axis for slice in (z, y, x) order (0=axial, 1=coronal, 2=sagittal)
    """
    fig = plt.figure(figsize=(20, 10))

    # Find best slices from crossed data (same slice for both volumes for consistent anatomy)
    if len(viz_cross.match_idx_a) > 0:
        matched_coords_a = viz_cross.coords_a[viz_cross.match_idx_a]
        if len(viz_cross.is_inlier) == len(matched_coords_a):
            inlier_coords_a = matched_coords_a[viz_cross.is_inlier]
        else:
            inlier_coords_a = np.array([]).reshape(0, 3)
        if len(inlier_coords_a) > 0:
            slice_idx = int(np.median(inlier_coords_a[:, slice_axis]))
        else:
            slice_idx = viz_cross.vol_a.shape[slice_axis] // 2
    else:
        slice_idx = viz_cross.vol_a.shape[slice_axis] // 2

    # Clamp to valid range
    slice_idx = int(np.clip(slice_idx, 0, viz_cross.vol_a.shape[slice_axis] - 1))

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.1)

    # Normalize volumes for display (misalignment applied slice-level for 2D visualization)
    vol_a_disp, vol_b_disp = _get_display_volumes(viz_cross)
    center_b = (np.array(viz_cross.vol_b.shape, dtype=np.float64) - 1.0) / 2.0

    # Top row: Without crossing
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])

    # Get slices from same anatomical position
    slice_a = get_slice(vol_a_disp, slice_axis, slice_idx)
    slice_b = get_slice(vol_b_disp, slice_axis, slice_idx)
    if viz_cross.is_transformed and viz_cross.rot_mat is not None and viz_cross.trans_vec is not None:
        slice_b = transform_slice_for_display(slice_b, viz_cross.rot_mat, viz_cross.trans_vec, center_b, slice_axis)

    window = 20

    # Get matched coordinates for no-crossing case (handle empty case)
    if len(viz_no_cross.match_idx_a) > 0:
        matched_coords_a_nc = viz_no_cross.coords_a[viz_no_cross.match_idx_a]
        matched_coords_b_nc_src = viz_no_cross.coords_a[viz_no_cross.match_idx_b]
        coords_a_2d_nc, _ = get_slice_coords(matched_coords_a_nc, slice_axis)
        coords_b_2d_nc_src, _ = get_slice_coords(matched_coords_b_nc_src, slice_axis)
        coords_b_2d_nc = coords_b_2d_nc_src
        if viz_no_cross.is_transformed and viz_no_cross.rot_mat is not None and viz_no_cross.trans_vec is not None:
            coords_b_2d_nc = transform_coords_for_display(
                coords_b_2d_nc_src, viz_no_cross.rot_mat, viz_no_cross.trans_vec, center_b, slice_axis
            )

        # Filter matches near the slice
        near_slice_nc = np.abs(matched_coords_a_nc[:, slice_axis] - slice_idx) < window
        coords_a_2d_nc = coords_a_2d_nc[near_slice_nc]
        coords_b_2d_nc = coords_b_2d_nc[near_slice_nc]
        if len(viz_no_cross.is_inlier) == len(viz_no_cross.match_idx_a):
            is_inlier_nc = viz_no_cross.is_inlier[near_slice_nc]
        else:
            is_inlier_nc = np.zeros(near_slice_nc.sum(), dtype=bool)
    else:
        coords_a_2d_nc = np.array([]).reshape(0, 2)
        coords_b_2d_nc = np.array([]).reshape(0, 2)
        is_inlier_nc = np.array([], dtype=bool)

    _plot_slice_with_matches(
        ax1, ax2, fig,
        slice_a, slice_b,
        coords_a_2d_nc, coords_b_2d_nc,
        is_inlier_nc,
        show_outliers=True,
        title_a=f"{viz_no_cross.modality_a.upper()} (No Crossing) slice {slice_idx}",
        title_b=f"{viz_no_cross.modality_b.upper()} slice {slice_idx}",
    )

    # Add metrics overlay
    num_inliers_nc = int(np.sum(viz_no_cross.is_inlier))
    ax1.text(
        0.02, 0.98,
        f"Inliers: {num_inliers_nc}\nIR: {viz_no_cross.inlier_ratio:.1%}\nTRE: {viz_no_cross.tre_mean:.2f}",
        transform=ax1.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
        color='white', fontweight='bold'
    )

    # Bottom row: With crossing
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax4 = fig.add_subplot(gs[1, 2:4])

    # Get matched coordinates for crossing case (handle empty case)
    if len(viz_cross.match_idx_a) > 0:
        matched_coords_a_c = viz_cross.coords_a[viz_cross.match_idx_a]
        matched_coords_b_c_src = viz_cross.coords_a[viz_cross.match_idx_b]
        coords_a_2d_c, _ = get_slice_coords(matched_coords_a_c, slice_axis)
        coords_b_2d_c_src, _ = get_slice_coords(matched_coords_b_c_src, slice_axis)
        coords_b_2d_c = coords_b_2d_c_src
        if viz_cross.is_transformed and viz_cross.rot_mat is not None and viz_cross.trans_vec is not None:
            coords_b_2d_c = transform_coords_for_display(
                coords_b_2d_c_src, viz_cross.rot_mat, viz_cross.trans_vec, center_b, slice_axis
            )

        # Filter matches near the slice
        near_slice_c = np.abs(matched_coords_a_c[:, slice_axis] - slice_idx) < window
        coords_a_2d_c = coords_a_2d_c[near_slice_c]
        coords_b_2d_c = coords_b_2d_c[near_slice_c]
        if len(viz_cross.is_inlier) == len(viz_cross.match_idx_a):
            is_inlier_c = viz_cross.is_inlier[near_slice_c]
        else:
            is_inlier_c = np.zeros(near_slice_c.sum(), dtype=bool)
    else:
        coords_a_2d_c = np.array([]).reshape(0, 2)
        coords_b_2d_c = np.array([]).reshape(0, 2)
        is_inlier_c = np.array([], dtype=bool)

    _plot_slice_with_matches(
        ax3, ax4, fig,
        slice_a, slice_b,
        coords_a_2d_c, coords_b_2d_c,
        is_inlier_c,
        show_outliers=True,
        title_a=f"{viz_cross.modality_a.upper()} (WITH Crossing) slice {slice_idx}",
        title_b=f"{viz_cross.modality_b.upper()} slice {slice_idx}",
    )

    # Add metrics overlay
    num_inliers_c = int(np.sum(viz_cross.is_inlier))
    ax3.text(
        0.02, 0.98,
        f"Inliers: {num_inliers_c}\nIR: {viz_cross.inlier_ratio:.1%}\nTRE: {viz_cross.tre_mean:.2f}",
        transform=ax3.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
        color='white', fontweight='bold'
    )

    # Title
    improvement = num_inliers_c - num_inliers_nc
    improvement_str = f"+{improvement}" if improvement >= 0 else str(improvement)
    view_names = {0: "Axial", 1: "Coronal", 2: "Sagittal"}
    view_name = view_names.get(slice_axis, "")
    misalign_info = f" [MISALIGNED rot={viz_cross.rotate_deg:.0f}°]" if viz_cross.is_transformed else ""
    fig.suptitle(
        f"{viz_cross.case_id}: Crossing Impact ({viz_cross.modality_a.upper()} <-> {viz_cross.modality_b.upper()}, "
        f"{view_name} slice {slice_idx}){misalign_info}\n"
        f"Without Crossing: {num_inliers_nc} inliers | "
        f"With Crossing: {num_inliers_c} inliers ({improvement_str})",
        fontsize=14, fontweight='bold'
    )

    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.close()


def plot_tta_comparison(
    viz_no_tta: VisualizationData,
    viz_with_tta: VisualizationData,
    tta_name: str,
    output_path: Path,
    slice_axis: int = 0,
) -> None:
    """
    Create a side-by-side comparison of matching with and without TTA.

    Args:
        viz_no_tta: Visualization data without TTA (crossing only)
        viz_with_tta: Visualization data with TTA
        tta_name: Name of the TTA adapter
        output_path: Path to save the figure
        slice_axis: Axis for slice in (z, y, x) order (0=axial, 1=coronal, 2=sagittal)
    """
    fig = plt.figure(figsize=(20, 10))

    # Find best slices from TTA data (same slice for both volumes for consistent anatomy)
    if len(viz_with_tta.match_idx_a) > 0:
        matched_coords_a = viz_with_tta.coords_a[viz_with_tta.match_idx_a]
        if len(viz_with_tta.is_inlier) == len(matched_coords_a):
            inlier_coords_a = matched_coords_a[viz_with_tta.is_inlier]
        else:
            inlier_coords_a = np.array([]).reshape(0, 3)
        if len(inlier_coords_a) > 0:
            slice_idx = int(np.median(inlier_coords_a[:, slice_axis]))
        else:
            slice_idx = viz_with_tta.vol_a.shape[slice_axis] // 2
    else:
        slice_idx = viz_with_tta.vol_a.shape[slice_axis] // 2

    slice_idx = int(np.clip(slice_idx, 0, viz_with_tta.vol_a.shape[slice_axis] - 1))

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.1)

    # Normalize volumes for display (misalignment applied slice-level for 2D visualization)
    vol_a_disp, vol_b_disp = _get_display_volumes(viz_with_tta)
    center_b = (np.array(viz_with_tta.vol_b.shape, dtype=np.float64) - 1.0) / 2.0

    # Get slices from same anatomical position
    slice_a = get_slice(vol_a_disp, slice_axis, slice_idx)
    slice_b = get_slice(vol_b_disp, slice_axis, slice_idx)
    if viz_with_tta.is_transformed and viz_with_tta.rot_mat is not None and viz_with_tta.trans_vec is not None:
        slice_b = transform_slice_for_display(
            slice_b, viz_with_tta.rot_mat, viz_with_tta.trans_vec, center_b, slice_axis
        )

    window = 20

    # Top row: Without TTA (crossing only)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])

    # Handle case where there are no matches
    if len(viz_no_tta.match_idx_a) > 0:
        matched_coords_a_nt = viz_no_tta.coords_a[viz_no_tta.match_idx_a]
        matched_coords_b_nt_src = viz_no_tta.coords_a[viz_no_tta.match_idx_b]
        coords_a_2d, _ = get_slice_coords(matched_coords_a_nt, slice_axis)
        coords_b_2d_src, _ = get_slice_coords(matched_coords_b_nt_src, slice_axis)
        coords_b_2d = coords_b_2d_src
        if viz_no_tta.is_transformed and viz_no_tta.rot_mat is not None and viz_no_tta.trans_vec is not None:
            coords_b_2d = transform_coords_for_display(
                coords_b_2d_src, viz_no_tta.rot_mat, viz_no_tta.trans_vec, center_b, slice_axis
            )

        # Filter matches near the slice
        near_slice = np.abs(matched_coords_a_nt[:, slice_axis] - slice_idx) < window
        coords_a_2d_f = coords_a_2d[near_slice]
        coords_b_2d_f = coords_b_2d[near_slice]
        if len(viz_no_tta.is_inlier) == len(viz_no_tta.match_idx_a):
            is_inlier_f = viz_no_tta.is_inlier[near_slice]
        else:
            is_inlier_f = np.zeros(near_slice.sum(), dtype=bool)
    else:
        coords_a_2d_f = np.array([]).reshape(0, 2)
        coords_b_2d_f = np.array([]).reshape(0, 2)
        is_inlier_f = np.array([], dtype=bool)

    _plot_slice_with_matches(
        ax1, ax2, fig,
        slice_a, slice_b,
        coords_a_2d_f, coords_b_2d_f,
        is_inlier_f,
        title_a=f"{viz_no_tta.modality_a.upper()} (Crossing Only) slice {slice_idx}",
        title_b=f"{viz_no_tta.modality_b.upper()} slice {slice_idx}",
    )

    num_inliers_no_tta = int(np.sum(viz_no_tta.is_inlier))
    ax1.text(
        0.02, 0.98,
        f"Inliers: {num_inliers_no_tta}\nIR: {viz_no_tta.inlier_ratio:.1%}\nTRE: {viz_no_tta.tre_mean:.2f}",
        transform=ax1.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor=COLORS["tta_disabled"], alpha=0.7),
        color='white', fontweight='bold'
    )

    # Bottom row: With TTA
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax4 = fig.add_subplot(gs[1, 2:4])

    # Handle case where there are no matches
    if len(viz_with_tta.match_idx_a) > 0:
        matched_coords_a_t = viz_with_tta.coords_a[viz_with_tta.match_idx_a]
        matched_coords_b_t_src = viz_with_tta.coords_a[viz_with_tta.match_idx_b]
        coords_a_2d, _ = get_slice_coords(matched_coords_a_t, slice_axis)
        coords_b_2d_src, _ = get_slice_coords(matched_coords_b_t_src, slice_axis)
        coords_b_2d = coords_b_2d_src
        if viz_with_tta.is_transformed and viz_with_tta.rot_mat is not None and viz_with_tta.trans_vec is not None:
            coords_b_2d = transform_coords_for_display(
                coords_b_2d_src, viz_with_tta.rot_mat, viz_with_tta.trans_vec, center_b, slice_axis
            )

        # Filter matches near the slice
        near_slice = np.abs(matched_coords_a_t[:, slice_axis] - slice_idx) < window
        coords_a_2d_f = coords_a_2d[near_slice]
        coords_b_2d_f = coords_b_2d[near_slice]
        if len(viz_with_tta.is_inlier) == len(viz_with_tta.match_idx_a):
            is_inlier_f = viz_with_tta.is_inlier[near_slice]
        else:
            is_inlier_f = np.zeros(near_slice.sum(), dtype=bool)
    else:
        coords_a_2d_f = np.array([]).reshape(0, 2)
        coords_b_2d_f = np.array([]).reshape(0, 2)
        is_inlier_f = np.array([], dtype=bool)

    _plot_slice_with_matches(
        ax3, ax4, fig,
        slice_a, slice_b,
        coords_a_2d_f, coords_b_2d_f,
        is_inlier_f,
        title_a=f"{viz_with_tta.modality_a.upper()} (Crossing + {tta_name}) slice {slice_idx}",
        title_b=f"{viz_with_tta.modality_b.upper()} slice {slice_idx}",
    )

    num_inliers_tta = int(np.sum(viz_with_tta.is_inlier))
    ax3.text(
        0.02, 0.98,
        f"Inliers: {num_inliers_tta}\nIR: {viz_with_tta.inlier_ratio:.1%}\nTRE: {viz_with_tta.tre_mean:.2f}",
        transform=ax3.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor=COLORS["tta_enabled"], alpha=0.7),
        color='white', fontweight='bold'
    )

    # Title
    view_names = {0: "Axial", 1: "Coronal", 2: "Sagittal"}
    view_name = view_names.get(slice_axis, "")
    improvement = num_inliers_tta - num_inliers_no_tta
    improvement_str = f"+{improvement}" if improvement >= 0 else str(improvement)
    fig.suptitle(
        f"{viz_with_tta.case_id}: TTA Impact ({tta_name}) "
        f"({viz_with_tta.modality_a.upper()} <-> {viz_with_tta.modality_b.upper()}, {view_name} slice {slice_idx})\n"
        f"Crossing Only: {num_inliers_no_tta} inliers | "
        f"Crossing + TTA: {num_inliers_tta} inliers ({improvement_str})",
        fontsize=14, fontweight='bold'
    )

    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.close()


def create_summary_figure(
    results: List[Dict],
    output_path: Path,
    title: str = "Evaluation Summary",
) -> None:
    """
    Create a summary figure showing aggregate metrics across all cases.

    Args:
        results: List of result dictionaries with metrics
        output_path: Path to save the figure
        title: Figure title
    """
    if not results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    case_ids = [r.get("case_id", f"Case{i}") for i, r in enumerate(results)]
    top1_vals = [r.get("top1_acc", 0) * 100 for r in results]
    inlier_ratios = [r.get("inlier_ratio", 0) * 100 for r in results]
    tre_means = [r.get("tre_mean", np.nan) for r in results]
    tre_means = [t if np.isfinite(t) else np.nan for t in tre_means]

    x = np.arange(len(case_ids))
    bar_width = 0.6

    # Top-1 accuracy
    ax = axes[0, 0]
    colors = ['green' if v >= 50 else 'orange' if v >= 20 else 'red' for v in top1_vals]
    ax.bar(x, top1_vals, bar_width, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(np.mean(top1_vals), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(top1_vals):.1f}%')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Top-1 Retrieval Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    # Inlier ratio
    ax = axes[0, 1]
    colors = ['green' if v >= 50 else 'orange' if v >= 20 else 'red' for v in inlier_ratios]
    ax.bar(x, inlier_ratios, bar_width, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(np.mean(inlier_ratios), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(inlier_ratios):.1f}%')
    ax.set_ylabel('Inlier Ratio (%)')
    ax.set_title('Inlier Ratio (TRE < 5 vox)')
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    # TRE distribution
    ax = axes[1, 0]
    valid_tre = [t for t in tre_means if np.isfinite(t)]
    if valid_tre:
        colors = ['green' if t <= 3 else 'orange' if t <= 5 else 'red' for t in valid_tre]
        ax.bar(range(len(valid_tre)), valid_tre, bar_width, color=colors, edgecolor='black', alpha=0.7)
        ax.axhline(np.mean(valid_tre), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(valid_tre):.2f} vox')
        ax.axhline(5.0, color='red', linestyle=':', linewidth=2, label='Inlier threshold')
    ax.set_ylabel('Mean TRE (voxels)')
    ax.set_title('Mean Target Registration Error')
    ax.set_xticks(range(len(valid_tre)))
    valid_ids = [case_ids[i] for i, t in enumerate(tre_means) if np.isfinite(t)]
    ax.set_xticklabels(valid_ids, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Summary stats text box
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    EVALUATION SUMMARY
    ==================

    Cases evaluated: {len(results)}

    Top-1 Accuracy:
      Mean: {np.mean(top1_vals):.1f}%
      Std:  {np.std(top1_vals):.1f}%
      Min:  {np.min(top1_vals):.1f}%
      Max:  {np.max(top1_vals):.1f}%

    Inlier Ratio:
      Mean: {np.mean(inlier_ratios):.1f}%
      Std:  {np.std(inlier_ratios):.1f}%

    Mean TRE (voxels):
      Mean: {np.nanmean(tre_means):.2f}
      Std:  {np.nanstd(tre_means):.2f}
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.close()
