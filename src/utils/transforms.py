"""Geometric transforms and augmentation utilities."""

from typing import Optional, Tuple, Union

import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation


def _validate_linear_transform_params(scale: float, shear_deg: float) -> None:
    if scale < 0:
        raise ValueError(f"scale must be >= 0, got {scale}")
    if scale >= 1.0:
        raise ValueError(f"scale must be < 1.0 (interpreted as ±fraction), got {scale}")
    if shear_deg < 0:
        raise ValueError(f"shear_deg must be >= 0, got {shear_deg}")
    if shear_deg >= 89.0:
        raise ValueError(f"shear_deg must be < 89 degrees to stay invertible, got {shear_deg}")


def generate_random_transform(
    rotate_deg: float,
    translate_mm: float,
    center: np.ndarray,
    rng: np.random.RandomState,
    voxel_sizes_mm: Optional[np.ndarray] = None,
    snap_translation_to_vox: bool = False,
    deterministic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate rotation matrix and translation vector.

    Args:
        rotate_deg: Rotation angle in degrees. If deterministic=False (default),
            this is the max angle and actual angles are sampled from
            [-rotate_deg, +rotate_deg] for each axis. If deterministic=True,
            this exact angle is used for all axes.
        translate_mm: Max translation in mm (symmetric range). If `voxel_sizes_mm`
            is provided, this is converted to voxels per axis.
        center: Center point for rotation
        rng: Random state for reproducibility
        voxel_sizes_mm: Optional voxel sizes (mm) for axes matching the coord order.
            If provided, translations are returned in voxel units.
        snap_translation_to_vox: If True, round translation to integer voxels.
        deterministic: If True, use rotate_deg as the exact angle for all axes
            instead of sampling. Use this when testing at specific rotation angles.

    Returns:
        Tuple of (rotation_matrix, translation_vector)
    """
    if rotate_deg < 0:
        raise ValueError(f"rotate_deg must be >= 0, got {rotate_deg}")
    if translate_mm < 0:
        raise ValueError(f"translate_mm must be >= 0, got {translate_mm}")

    if deterministic:
        angles = np.array([rotate_deg, rotate_deg, rotate_deg], dtype=np.float64)
    else:
        angles = rng.uniform(-rotate_deg, rotate_deg, size=3)
    r = Rotation.from_euler('xyz', angles, degrees=True)
    rot_mat = r.as_matrix()

    if voxel_sizes_mm is None:
        trans = rng.uniform(-translate_mm, translate_mm, size=3)
    else:
        voxel_sizes_mm = np.asarray(voxel_sizes_mm, dtype=np.float64).reshape(3)
        max_trans_vox = float(translate_mm) / (voxel_sizes_mm + 1e-8)
        trans = rng.uniform(-max_trans_vox, max_trans_vox, size=3)

    if snap_translation_to_vox:
        trans = np.round(trans)

    return rot_mat, trans


def generate_random_transform_in_plane(
    rotate_deg: float,
    translate_mm: float,
    center: np.ndarray,
    rng: np.random.RandomState,
    voxel_sizes_mm: Optional[np.ndarray] = None,
    snap_translation_to_vox: bool = False,
    fixed_axis: int = 0,
    deterministic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a rigid transform constrained to a single slice plane.

    This is intended for strictly-2D descriptors (e.g., axial-slice SIFT) where we
    must not mix information across slices.

    Args:
        rotate_deg: Rotation angle in degrees. If deterministic=False (default),
            this is the max angle and the actual angle is sampled from
            [-rotate_deg, +rotate_deg]. If deterministic=True, this exact angle
            is used.
        translate_mm: Max translation in mm. If voxel_sizes_mm is provided, this
            is converted to voxels per axis.
        center: Center point for rotation.
        rng: Random state for reproducibility.
        voxel_sizes_mm: Optional voxel sizes (mm) for axes matching the coord order.
        snap_translation_to_vox: If True, round translation to integer voxels.
        fixed_axis: Which axis is fixed (0=axial, 1=coronal, 2=sagittal).
        deterministic: If True, use rotate_deg as the exact angle instead of
            sampling from [-rotate_deg, +rotate_deg]. Use this when testing at
            specific rotation angles (e.g., degradation curves).

    Conventions:
    - Coordinates are (z, y, x)
    - fixed_axis=0 => axial plane (z fixed), rotate/translate in (y, x)
    - fixed_axis=1 => coronal plane (y fixed), rotate/translate in (z, x)
    - fixed_axis=2 => sagittal plane (x fixed), rotate/translate in (z, y)
    """
    if rotate_deg < 0:
        raise ValueError(f"rotate_deg must be >= 0, got {rotate_deg}")
    if translate_mm < 0:
        raise ValueError(f"translate_mm must be >= 0, got {translate_mm}")
    if fixed_axis not in (0, 1, 2):
        raise ValueError(f"fixed_axis must be 0, 1, or 2; got {fixed_axis}")

    # Pick a single in-plane rotation angle
    if deterministic:
        angle_deg = float(rotate_deg)
    else:
        angle_deg = float(rng.uniform(-rotate_deg, rotate_deg))
    theta = np.deg2rad(angle_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    rot_mat = np.eye(3, dtype=np.float64)
    plane_axes = [0, 1, 2]
    plane_axes.remove(fixed_axis)
    ax0, ax1 = plane_axes  # rotate in (ax0, ax1) plane
    rot_mat[ax0, ax0] = c
    rot_mat[ax0, ax1] = -s
    rot_mat[ax1, ax0] = s
    rot_mat[ax1, ax1] = c

    # Translation in voxels (or mm if voxel_sizes_mm is None), but never along fixed axis
    if voxel_sizes_mm is None:
        max_trans = np.array([translate_mm, translate_mm, translate_mm], dtype=np.float64)
    else:
        voxel_sizes_mm = np.asarray(voxel_sizes_mm, dtype=np.float64).reshape(3)
        max_trans = float(translate_mm) / (voxel_sizes_mm + 1e-8)

    trans = rng.uniform(-max_trans, max_trans, size=3).astype(np.float64, copy=False)
    trans[fixed_axis] = 0.0

    if snap_translation_to_vox:
        trans = np.round(trans)

    return rot_mat, trans


def generate_random_affine_in_plane(
    rotate_deg: float,
    translate_mm: float,
    scale: float,
    shear_deg: float,
    center: np.ndarray,
    rng: np.random.RandomState,
    voxel_sizes_mm: Optional[np.ndarray] = None,
    snap_translation_to_vox: bool = False,
    fixed_axis: int = 0,
    deterministic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random in-plane affine transform (rotation + translation + scale + shear).

    Forward convention (feature motion):
        out = A @ (in - center) + center + t

    Notes:
    - This is intended for 2D (slice-based) descriptors. Use `fixed_axis=0` for axial slices
      with coordinate order (z, y, x).
    - `scale` is interpreted as a max fractional deviation, i.e. per-axis factors are sampled
      uniformly from [1-scale, 1+scale]. Requires `0 <= scale < 1`.
    - `shear_deg` is a max shear angle (in degrees). Shear is applied as x += k*y (in-plane),
      where k = tan(shear_angle).
    """
    _validate_linear_transform_params(scale=float(scale), shear_deg=float(shear_deg))

    rot_mat, trans = generate_random_transform_in_plane(
        rotate_deg=rotate_deg,
        translate_mm=translate_mm,
        center=center,
        rng=rng,
        voxel_sizes_mm=voxel_sizes_mm,
        snap_translation_to_vox=snap_translation_to_vox,
        fixed_axis=fixed_axis,
        deterministic=deterministic,
    )

    if float(scale) == 0.0 and float(shear_deg) == 0.0:
        return rot_mat, trans

    plane_axes = [0, 1, 2]
    plane_axes.remove(int(fixed_axis))
    ax0, ax1 = plane_axes

    # Per-axis scale factors within the in-plane axes (anisotropic by default).
    if float(scale) > 0.0:
        s0 = float(rng.uniform(1.0 - float(scale), 1.0 + float(scale)))
        s1 = float(rng.uniform(1.0 - float(scale), 1.0 + float(scale)))
    else:
        s0, s1 = 1.0, 1.0

    if float(shear_deg) > 0.0:
        sh_angle = float(rng.uniform(-float(shear_deg), float(shear_deg)))
        k = float(np.tan(np.deg2rad(sh_angle)))
    else:
        k = 0.0

    # In-plane scale+shear matrix in the (ax0, ax1) coordinate order.
    # Shear is applied as: axis1 += k * axis0  (e.g., x += k*y for axial slices).
    lin = np.eye(3, dtype=np.float64)
    lin[ax0, ax0] = s0
    lin[ax1, ax0] = s1 * k
    lin[ax1, ax1] = s1

    return rot_mat @ lin, trans


def apply_transform_to_coords(
    coords: np.ndarray,
    rot_mat: np.ndarray,
    trans: np.ndarray,
    center: np.ndarray
) -> np.ndarray:
    """Apply rotation (around center) and translation to coordinates.

    Args:
        coords: Coordinates to transform (N, 3)
        rot_mat: 3x3 rotation matrix
        trans: Translation vector (3,)
        center: Center point for rotation

    Returns:
        Transformed coordinates (N, 3)
    """
    centered = coords - center
    rotated = (rot_mat @ centered.T).T
    transformed = rotated + center + trans
    return transformed


def transform_keypoints_with_z(
    keypoints_with_z: list[tuple[int, object]],
    lin_mat: np.ndarray,
    trans: np.ndarray,
    center: np.ndarray,
    *,
    keep_upright: bool = False,
    update_size: bool = True,
    allow_z_change: bool = False,
    min_size: float = 1.0,
    max_size: float = 1e4,
) -> list[tuple[int, object]]:
    """
    Transform cv2.KeyPoint geometry (x, y, size, angle) using a known (z, y, x) transform.

    This is used to keep correspondences when one modality is warped (asymmetric augmentation
    or test-time misalignment simulation).

    Args:
        keypoints_with_z: List of (z, cv2.KeyPoint).
        lin_mat/trans/center: Define the forward mapping:
            out = A @ (in - center) + center + t
        keep_upright: If True, output keypoint angles are set to 0 (upright SIFT).
        update_size: If True, scale `kp.size` by sqrt(|det(A_xy)|) where A_xy is the in-plane
            linear transform in (x, y).
        allow_z_change: If False (default), requires transformed z to round back to input z.
        min_size/max_size: Clamp output keypoint size to this range.
    """
    if not keypoints_with_z:
        return []

    # Import lazily to avoid hard dependency for non-SIFT users.
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover - import-time dependent
        raise ImportError("OpenCV is required to transform cv2.KeyPoint objects.") from e

    lin_mat = np.asarray(lin_mat, dtype=np.float64).reshape(3, 3)
    trans = np.asarray(trans, dtype=np.float64).reshape(3)
    center = np.asarray(center, dtype=np.float64).reshape(3)

    # For axial slices (z fixed), OpenCV keypoints use (x, y) = (axis 2, axis 1).
    a_xy = np.array(
        [
            [lin_mat[2, 2], lin_mat[2, 1]],
            [lin_mat[1, 2], lin_mat[1, 1]],
        ],
        dtype=np.float64,
    )

    scale_factor = 1.0
    if update_size:
        det = float(np.linalg.det(a_xy))
        scale_factor = float(np.sqrt(abs(det)))
        if not np.isfinite(scale_factor) or scale_factor <= 0.0:
            scale_factor = 1.0

    coords = np.array(
        [
            [float(z), float(kp.pt[1]), float(kp.pt[0])]
            for z, kp in keypoints_with_z
        ],
        dtype=np.float64,
    )
    coords_t = apply_transform_to_coords(coords, lin_mat, trans, center)

    out: list[tuple[int, object]] = []
    for (z_in, kp), (z_t, y_t, x_t) in zip(keypoints_with_z, coords_t):
        z_out = int(np.round(float(z_t)))
        if not allow_z_change and int(z_out) != int(z_in):
            continue

        angle_in = float(getattr(kp, "angle", 0.0))
        if keep_upright:
            angle_out = 0.0
        else:
            a = float(np.deg2rad(angle_in))
            v = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
            v_t = a_xy @ v
            if float(np.linalg.norm(v_t)) > 0.0:
                angle_out = float(np.rad2deg(np.arctan2(v_t[1], v_t[0]))) % 360.0
            else:
                angle_out = angle_in

        size_in = float(getattr(kp, "size", 0.0))
        size_out = size_in * scale_factor if update_size else size_in
        size_out = float(np.clip(size_out, float(min_size), float(max_size)))

        response = float(getattr(kp, "response", 0.0))
        octave = int(getattr(kp, "octave", 0))
        class_id = int(getattr(kp, "class_id", -1))

        out.append(
            (
                z_out,
                cv2.KeyPoint(float(x_t), float(y_t), float(size_out), float(angle_out), response, octave, class_id),
            )
        )

    return out


def warp_volume_rigid(
    volume: np.ndarray,
    rot_mat: np.ndarray,
    trans_vox: Union[np.ndarray, Tuple[float, float, float]],
    *,
    center: Optional[np.ndarray] = None,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """
    Apply a rigid transform to a 3D volume and resample onto the same grid.

    Forward convention (feature motion):
        out = R @ (in - center) + center + t

    `scipy.ndimage.affine_transform` expects the inverse mapping (out -> in):
        in = R^T @ (out - center - t) + center

    Args:
        volume: 3D volume to transform (shape (D, H, W))
        rot_mat: 3x3 rotation matrix (acts on [z, y, x] coords)
        trans_vox: Translation vector in voxel units (3,)
        center: Center of rotation in voxel coords; defaults to (shape-1)/2
        order: Interpolation order (0=nearest, 1=linear)
        cval: Constant value for samples outside the input volume

    Returns:
        Transformed volume (float32)
    """
    if volume.ndim != 3:
        raise ValueError(f"warp_volume_rigid expects a 3D volume, got shape={volume.shape}")

    shape = np.asarray(volume.shape, dtype=np.float64)
    if center is None:
        center = (shape - 1.0) / 2.0
    center = np.asarray(center, dtype=np.float64).reshape(3)
    trans_vox = np.asarray(trans_vox, dtype=np.float64).reshape(3)

    rot_inv = rot_mat.T
    offset = center - rot_inv @ center - rot_inv @ trans_vox

    warped = affine_transform(
        volume.astype(np.float32, copy=False),
        rot_inv,
        offset=offset,
        output_shape=tuple(int(s) for s in volume.shape),
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=(order > 1),
    )
    return warped.astype(np.float32, copy=False)


def warp_volume_affine(
    volume: np.ndarray,
    lin_mat: np.ndarray,
    trans_vox: Union[np.ndarray, Tuple[float, float, float]],
    *,
    center: Optional[np.ndarray] = None,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """
    Apply a general (invertible) linear transform + translation to a 3D volume.

    Forward convention (feature motion):
        out = A @ (in - center) + center + t

    `scipy.ndimage.affine_transform` expects the inverse mapping (out -> in):
        in = A^{-1} @ (out - center - t) + center
    """
    if volume.ndim != 3:
        raise ValueError(f"warp_volume_affine expects a 3D volume, got shape={volume.shape}")

    shape = np.asarray(volume.shape, dtype=np.float64)
    if center is None:
        center = (shape - 1.0) / 2.0
    center = np.asarray(center, dtype=np.float64).reshape(3)
    trans_vox = np.asarray(trans_vox, dtype=np.float64).reshape(3)
    lin_mat = np.asarray(lin_mat, dtype=np.float64).reshape(3, 3)

    det = float(np.linalg.det(lin_mat))
    if not np.isfinite(det) or abs(det) < 1e-8:
        raise ValueError(f"lin_mat must be invertible; det={det}")

    lin_inv = np.linalg.inv(lin_mat)
    offset = center - lin_inv @ center - lin_inv @ trans_vox

    warped = affine_transform(
        volume.astype(np.float32, copy=False),
        lin_inv,
        offset=offset,
        output_shape=tuple(int(s) for s in volume.shape),
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=(order > 1),
    )
    return warped.astype(np.float32, copy=False)
