"""NIfTI orientation utilities.

Ensures volumes are loaded with axis 0 corresponding to the axial (I/S)
direction, regardless of the on-disk storage orientation.

ReMIND stores volumes as (I, P, L) so axis 0 is already axial.
BraTS stores as (L, P, S) and RESECT as (R, A, S) — axis 2 is axial.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_axial_axis(affine: np.ndarray) -> int:
    """Return the array axis corresponding to the axial (I/S) direction.

    Uses nibabel orientation detection on the NIfTI affine matrix.
    Falls back to axis 0 if no I/S axis is found (should not happen
    for brain MRI).
    """
    import nibabel as nib

    ornt = nib.orientations.io_orientation(affine)
    labels = nib.orientations.ornt2axcodes(ornt)
    for i, code in enumerate(labels):
        if code in ("I", "S"):
            return i
    logger.warning(
        "No I/S axis found in orientation %s; falling back to axis 0", labels,
    )
    return 0


def get_axial_dim(affine: np.ndarray, shape: tuple[int, ...]) -> int:
    """Return the size of the axial (I/S) dimension."""
    return shape[get_axial_axis(affine)]


def reorient_to_axial_first(
    vol: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorient a 3D volume so axis 0 = axial (I/S) direction.

    If axis 0 is already axial, returns the inputs unchanged (no copy).

    Returns:
        (reoriented_volume, reoriented_affine)
    """
    axis = get_axial_axis(affine)
    if axis == 0:
        return vol, affine

    # Build permutation: move axial axis to position 0, keep others in order
    axes = list(range(vol.ndim))
    axes.remove(axis)
    axes.insert(0, axis)

    vol = np.transpose(vol, axes)

    # Update affine: permute columns to match new axis order
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, axes]

    return vol, new_affine
