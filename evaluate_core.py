#!/usr/bin/env python3
"""Internal matching and metric implementation for :mod:`evaluate`.

Public users should invoke ``python evaluate.py --help`` rather than executing
this implementation module directly.
"""

import argparse
import hashlib
import json
import logging
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import cdist

from src.adapter import BaseAdapter, InlierRidgeAdapter
from src.baselines.defaults import add_baseline_args
from src.baselines.nocross import NoCrossBaselineRunner
from src.cross.models import DisentangledVAECrosser
from src.eval.metrics import compute_sr_auc_from_tre
from src.eval.types import EvaluationResult, make_error_result as _make_error_result
from src.descriptor import SIFTDescriptor
from src.io.case_loader import CaseData
from src.io.dataset_registry import build_manifest
from src.io.case_adapter import load_cases_from_manifest
from src.io.modality_aliases import find_modality_index, get_equivalents, resolve_modality_alias
from src.extraction import (
    _extract_paired_from_keypoints,
    _extract_paired_from_keypoints_sets,
    _extract_with_keypoint_geometry,
)
from src.utils.sampling import (
    _infer_signal_mask,
    ordered_unique_ints,
    sample_coords_sift_union_from_roi,
    sample_keypoints_2d_dual,
    sample_keypoints_2d_ref,
    select_slices_by_roi_area,
    select_slices_uniform_random,
)
from src.utils.roi_preprocessing import compute_safe_crop_bounds, safe_preprocess_roi
from src.utils.transforms import (
    apply_transform_to_coords as apply_transform_to_coords_vox,
    generate_random_affine_in_plane as generate_random_affine_in_plane_vox,
    generate_random_transform as generate_random_transform_vox,
    generate_random_transform_in_plane as generate_random_transform_in_plane_vox,
    transform_keypoints_with_z,
    warp_volume_rigid,
    warp_volume_affine,
)
from src.viz import (
    VisualizationData,
    plot_matches_multiview,
    plot_tre_distribution,
    plot_crossing_comparison,
    plot_tta_comparison,
    create_summary_figure,
)

logger = logging.getLogger(__name__)


def _create_nocross_baseline(args: argparse.Namespace) -> tuple[Optional[Any], Optional[str]]:
    baseline = str(getattr(args, "baseline", "none")).lower()
    if baseline == "none":
        return None, None
    if baseline == "nocross":
        return NoCrossBaselineRunner(), None
    return None, f"Unsupported baseline in CrossFeat: {baseline}"


def _display_id(case_id: str) -> str:
    """Extract short display ID from a case_id (last path segment).

    For path-based IDs like "sun/test/MAP_10_point1/000050" returns "000050".
    For already-short IDs like "Case013:z042" or "285" returns them unchanged.
    """
    return case_id.rsplit("/", 1)[-1] if "/" in case_id else case_id


@dataclass
class _PreparedCaseData:
    """Internal dataclass for prepared case data before crossing/matching."""
    case_id: str
    mod_a: str
    mod_b: str
    vol_a: np.ndarray
    vol_b: np.ndarray
    coords_a: np.ndarray
    coords_b: np.ndarray
    d_a: np.ndarray  # PCA-transformed descriptors
    d_b: np.ndarray  # PCA-transformed descriptors
    rot_mat_viz: Optional[np.ndarray]
    trans_vec_viz: Optional[np.ndarray]
    rotate_deg: float
    translate_mm: float
    affine_scale: float
    affine_shear_deg: float
    two_d_only: bool


def _prepare_case_data(
    case: CaseData,
    modalities: List[str],
    extractor,
    pca,
    rotate_deg: float,
    translate_mm: float,
    affine_scale: float,
    affine_shear_deg: float,
    n_samples: int,
    seed: int,
    two_d_only: bool,
    pair_sampling: str,
    deterministic_transform: bool,
    max_kpts_per_slice: int = 500,
    z_slices: Optional[List[int]] = None,
) -> Union[_PreparedCaseData, Tuple[None, str]]:
    """
    Prepare case data for evaluation: sample coords, extract descriptors, apply PCA.

    Returns:
        _PreparedCaseData on success, or (None, error_message) on failure.
    """
    case_id = case.case_id
    mod_a, mod_b = modalities[0], modalities[1]
    vol_a = case.volumes[mod_a]
    vol_b = case.volumes[mod_b]

    # Resize to common (H, W) when modalities have different spatial dims
    # (e.g. MS2 RGB 384x1224 vs NIR 352x1280 from different cameras).
    if vol_a.shape != vol_b.shape:
        import cv2

        min_h = min(vol_a.shape[1], vol_b.shape[1])
        min_w = min(vol_a.shape[2], vol_b.shape[2])
        if vol_a.shape[1:] != (min_h, min_w):
            vol_a = cv2.resize(
                vol_a[0], (min_w, min_h), interpolation=cv2.INTER_AREA,
            )[np.newaxis, :, :].astype(np.float32)
        if vol_b.shape[1:] != (min_h, min_w):
            vol_b = cv2.resize(
                vol_b[0], (min_w, min_h), interpolation=cv2.INTER_AREA,
            )[np.newaxis, :, :].astype(np.float32)

    rng = np.random.RandomState(seed)

    base_margin = 15
    erode_margin = 5

    # Joint ROI mask in reference space (before any simulated misalignment).
    abs_a = np.abs(vol_a)
    abs_b = np.abs(vol_b)
    max_a = float(np.max(abs_a))
    max_b = float(np.max(abs_b))
    if max_a <= 0.0 or max_b <= 0.0:
        return None, "No signal in one or both modalities"

    # Use a robust "signal" mask to avoid sampling from constant background values
    # (e.g. -1 after normalization) which can otherwise pass the relative threshold.
    mask_a = _infer_signal_mask(vol_a, abs_volume=abs_a, max_abs=max_a)
    mask_b = _infer_signal_mask(vol_b, abs_volume=abs_b, max_abs=max_b)
    # Apply a relative threshold as well to suppress low-amplitude non-background noise.
    mask_a &= abs_a > 0.01 * max_a
    mask_b &= abs_b > 0.01 * max_b
    mask = mask_a & mask_b
    mask = safe_preprocess_roi(mask, erode_iterations=erode_margin, base_margin=base_margin)

    z, y, x = np.where(mask)
    shape = np.array(vol_a.shape)
    coords_pool = np.stack([z, y, x], axis=1).astype(np.int32, copy=False)
    if len(coords_pool) < 100:
        return None, "Not enough joint ROI coordinates"

    n_goal = int(min(n_samples, len(coords_pool)))
    crop_min, crop_max = compute_safe_crop_bounds(tuple(int(s) for s in shape), base_margin)

    if pair_sampling in {"2d_sift_ref", "2d_sift_dual"}:
        if not hasattr(extractor, "compute_2d") or not hasattr(extractor, "detect_keypoints_2d"):
            return None, f"pair_sampling={pair_sampling!r} requires a SIFT extractor with 2D keypoint APIs"
        if not two_d_only and (rotate_deg > 0 or translate_mm > 0 or affine_scale > 0 or affine_shear_deg > 0):
            return None, f"pair_sampling={pair_sampling!r} requires two_d_only=True for transforms (slice-based SIFT)"

        max_kpts_per_slice = int(max_kpts_per_slice)
        if max_kpts_per_slice <= 0:
            return None, f"max_kpts_per_slice must be > 0, got {max_kpts_per_slice}"

        keep_upright = bool(getattr(extractor, "compute_orientation", True)) is False

        # Generate transform for misalignment simulation (applied to modality B).
        center = (np.array(vol_a.shape) - 1.0) / 2.0
        rot_mat_viz = None
        trans_vec_viz = None
        vol_b_eval = vol_b
        do_transform = rotate_deg > 0 or translate_mm > 0 or affine_scale > 0 or affine_shear_deg > 0
        lin_mat = np.eye(3, dtype=np.float64)
        trans = np.zeros(3, dtype=np.float64)
        if do_transform:
            affine_b = case.affines.get(mod_b)
            affine = affine_b if affine_b is not None else case.affines.get(mod_a)
            voxel_sizes_mm = None
            if affine is not None:
                voxel_sizes_mm = np.sqrt(np.sum(np.square(affine[:3, :3]), axis=0))

            if (affine_scale > 0 or affine_shear_deg > 0) and not two_d_only:
                return None, "Affine misalignment currently requires two_d_only=True"

            if affine_scale > 0 or affine_shear_deg > 0:
                lin_mat, trans = generate_random_affine_in_plane_vox(
                    rotate_deg=rotate_deg,
                    translate_mm=translate_mm,
                    scale=affine_scale,
                    shear_deg=affine_shear_deg,
                    center=center,
                    rng=rng,
                    voxel_sizes_mm=voxel_sizes_mm,
                    snap_translation_to_vox=True,
                    fixed_axis=0,  # axial plane
                    deterministic=deterministic_transform,
                )
                vol_b_eval = warp_volume_affine(vol_b, lin_mat, trans, center=center, order=1, cval=0.0)
            else:
                gen = generate_random_transform_in_plane_vox if two_d_only else generate_random_transform_vox
                gen_kwargs = {
                    "rotate_deg": rotate_deg,
                    "translate_mm": translate_mm,
                    "center": center,
                    "rng": rng,
                    "voxel_sizes_mm": voxel_sizes_mm,
                    "snap_translation_to_vox": True,
                    "deterministic": deterministic_transform,
                }
                if two_d_only:
                    gen_kwargs["fixed_axis"] = 0  # axial plane
                lin_mat, trans = gen(**gen_kwargs)
                vol_b_eval = warp_volume_rigid(vol_b, lin_mat, trans, center=center, order=1, cval=0.0)

            rot_mat_viz = lin_mat
            trans_vec_viz = trans

        def _kps_to_coords(kps_with_z: list[tuple[int, Any]]) -> np.ndarray:
            if not kps_with_z:
                return np.zeros((0, 3), dtype=np.int32)
            coords = np.array(
                [
                    # OpenCV KeyPoint.pt is (x, y); repo convention is (z, y, x).
                    [int(z), int(np.round(kp.pt[1])), int(np.round(kp.pt[0]))]
                    for z, kp in kps_with_z
                ],
                dtype=np.int32,
            )
            return coords

        mask_b_eval = None
        if do_transform:
            abs_b_eval = np.abs(vol_b_eval)
            max_b_eval = float(np.max(abs_b_eval))
            if max_b_eval <= 0.0:
                return None, "No signal in warped modality B"
            mask_b_eval = _infer_signal_mask(vol_b_eval, abs_volume=abs_b_eval, max_abs=max_b_eval)
            mask_b_eval &= abs_b_eval > 0.01 * max_b_eval
            mask_b_eval = safe_preprocess_roi(mask_b_eval, erode_iterations=erode_margin, base_margin=base_margin)

        def _filter_aligned_keypoint_pairs(
            kps_a: list[tuple[int, Any]],
            kps_b: list[tuple[int, Any]],
        ) -> tuple[list[tuple[int, Any]], list[tuple[int, Any]], np.ndarray, np.ndarray]:
            if not kps_a or not kps_b:
                return [], [], np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.int32)
            if len(kps_a) != len(kps_b):
                raise ValueError("Aligned keypoint lists must have the same length.")

            coords_a = _kps_to_coords(kps_a)
            coords_b = _kps_to_coords(kps_b)

            # IMPORTANT: avoid indexing masks with out-of-bounds coords.
            # Even if `keep` would later discard those points, advanced indexing
            # evaluates first and can raise IndexError.
            n = len(coords_a)
            keep = np.ones(n, dtype=bool)

            shape_a = np.asarray(shape, dtype=np.int64).reshape(3)
            shape_b = np.asarray(vol_b_eval.shape, dtype=np.int64).reshape(3) if do_transform else shape_a

            in_bounds_a = np.all((coords_a >= 0) & (coords_a < shape_a), axis=1)
            in_bounds_b = np.all((coords_b >= 0) & (coords_b < shape_b), axis=1)
            keep &= in_bounds_a & in_bounds_b

            mask_a_ok = np.zeros(n, dtype=bool)
            if bool(np.any(in_bounds_a)):
                idx_a = np.where(in_bounds_a)[0]
                ca = coords_a[idx_a]
                mask_a_ok[idx_a] = mask[ca[:, 0], ca[:, 1], ca[:, 2]]
            keep &= mask_a_ok

            if do_transform and mask_b_eval is not None:
                cb_crop_min, cb_crop_max = compute_safe_crop_bounds(tuple(int(s) for s in shape_b), base_margin)
                keep &= np.all((coords_b >= cb_crop_min) & (coords_b < cb_crop_max), axis=1)
                mask_b_ok = np.zeros(n, dtype=bool)
                if bool(np.any(in_bounds_b)):
                    idx_b = np.where(in_bounds_b)[0]
                    cb = coords_b[idx_b]
                    mask_b_ok[idx_b] = mask_b_eval[cb[:, 0], cb[:, 1], cb[:, 2]]
                keep &= mask_b_ok

            idx = np.where(keep)[0].tolist()
            kps_a_f = [kps_a[i] for i in idx]
            kps_b_f = [kps_b[i] for i in idx]
            return kps_a_f, kps_b_f, coords_a[keep], coords_b[keep]

        def _extract_one_set(
            kps_a: list[tuple[int, Any]],
            kps_b: list[tuple[int, Any]],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            kps_a_f, kps_b_f, coords_a_f, coords_b_f = _filter_aligned_keypoint_pairs(kps_a, kps_b)
            if len(kps_a_f) == 0:
                empty = np.zeros((0, int(getattr(extractor, "dim", 128))), dtype=np.float32)
                return empty, empty, coords_a_f, coords_b_f

            if do_transform:
                d_a_raw, d_b_raw = _extract_paired_from_keypoints_sets(extractor, vol_a, vol_b_eval, kps_a_f, kps_b_f)
            else:
                d_a_raw, d_b_raw = _extract_paired_from_keypoints(extractor, vol_a, vol_b, kps_a_f)

            valid_desc = (np.linalg.norm(d_a_raw, axis=1) > 0.1) & (np.linalg.norm(d_b_raw, axis=1) > 0.1)
            return d_a_raw[valid_desc], d_b_raw[valid_desc], coords_a_f[valid_desc], coords_b_f[valid_desc]

        if pair_sampling == "2d_sift_ref":
            kps_a = sample_keypoints_2d_ref(
                extractor,
                vol_ref=vol_a,
                roi_mask=mask,
                crop_min=crop_min,
                crop_max=crop_max,
                n_goal=n_goal,
                max_keypoints_per_slice=max_kpts_per_slice,
                rng=rng,
                z_slices=z_slices,
            )
            if not kps_a:
                return None, "No keypoints detected in ROI"

            kps_b = (
                transform_keypoints_with_z(
                    kps_a,
                    lin_mat,
                    trans,
                    center,
                    keep_upright=keep_upright,
                    update_size=(affine_scale > 0 or affine_shear_deg > 0),
                    allow_z_change=False,
                )
                if do_transform
                else kps_a
            )
            d_a_raw, d_b_raw, coords_a, coords_b = _extract_one_set(kps_a, kps_b)
        else:
            kps_a_anchor, kps_b_anchor = sample_keypoints_2d_dual(
                extractor,
                vol_a=vol_a,
                vol_b=vol_b,
                roi_mask=mask,
                crop_min=crop_min,
                crop_max=crop_max,
                n_goal=n_goal,
                max_keypoints_per_slice=max_kpts_per_slice,
                rng=rng,
                z_slices=z_slices,
            )
            if not kps_a_anchor and not kps_b_anchor:
                return None, "No keypoints detected in ROI"

            kps_a_anchor_b = (
                transform_keypoints_with_z(
                    kps_a_anchor,
                    lin_mat,
                    trans,
                    center,
                    keep_upright=keep_upright,
                    update_size=(affine_scale > 0 or affine_shear_deg > 0),
                    allow_z_change=False,
                )
                if do_transform
                else kps_a_anchor
            )
            kps_b_anchor_b = (
                transform_keypoints_with_z(
                    kps_b_anchor,
                    lin_mat,
                    trans,
                    center,
                    keep_upright=keep_upright,
                    update_size=(affine_scale > 0 or affine_shear_deg > 0),
                    allow_z_change=False,
                )
                if do_transform
                else kps_b_anchor
            )

            d_a1, d_b1, coords_a1, coords_b1 = _extract_one_set(kps_a_anchor, kps_a_anchor_b)
            d_a2, d_b2, coords_a2, coords_b2 = _extract_one_set(kps_b_anchor, kps_b_anchor_b)

            parts_a = [p for p in (d_a1, d_a2) if p.size != 0]
            parts_b = [p for p in (d_b1, d_b2) if p.size != 0]
            parts_ca = [p for p in (coords_a1, coords_a2) if p.size != 0]
            parts_cb = [p for p in (coords_b1, coords_b2) if p.size != 0]

            if parts_a:
                d_a_raw = np.vstack(parts_a)
                d_b_raw = np.vstack(parts_b)
                coords_a = np.vstack(parts_ca)
                coords_b = np.vstack(parts_cb)
            else:
                d_a_raw = np.zeros((0, int(getattr(extractor, "dim", 128))), dtype=np.float32)
                d_b_raw = np.zeros((0, int(getattr(extractor, "dim", 128))), dtype=np.float32)
                coords_a = np.zeros((0, 3), dtype=np.int32)
                coords_b = np.zeros((0, 3), dtype=np.int32)

        valid_desc = (np.linalg.norm(d_a_raw, axis=1) > 0.1) & (np.linalg.norm(d_b_raw, axis=1) > 0.1)
        d_a_raw = d_a_raw[valid_desc]
        d_b_raw = d_b_raw[valid_desc]
        coords_a = coords_a[valid_desc]
        coords_b = coords_b[valid_desc]

        if len(d_a_raw) < 32:
            return None, "Too few valid descriptors"

        if pca is not None:
            d_a = pca.transform(d_a_raw).astype(np.float32)
            d_b = pca.transform(d_b_raw).astype(np.float32)
        else:
            d_a = d_a_raw.astype(np.float32)
            d_b = d_b_raw.astype(np.float32)

        return _PreparedCaseData(
            case_id=case_id,
            mod_a=mod_a,
            mod_b=mod_b,
            vol_a=vol_a,
            vol_b=vol_b,
            coords_a=coords_a,
            coords_b=coords_b,
            d_a=d_a,
            d_b=d_b,
            rot_mat_viz=rot_mat_viz,
            trans_vec_viz=trans_vec_viz,
            rotate_deg=rotate_deg,
            translate_mm=translate_mm,
            affine_scale=affine_scale,
            affine_shear_deg=affine_shear_deg,
            two_d_only=two_d_only,
        )

    if pair_sampling == "sift_union":
        coords_a = sample_coords_sift_union_from_roi(
            extractor,
            vol_a=vol_a,
            vol_b=vol_b,
            roi_mask=mask,
            crop_min=crop_min,
            crop_max=crop_max,
            coords_pool=coords_pool,
            n_goal=n_goal,
            rng=rng,
        )
    elif pair_sampling in {"random", "random_with_kp"}:
        idx = rng.choice(len(coords_pool), size=n_goal, replace=False)
        coords_a = coords_pool[idx]
    else:
        return None, f"Unknown pair_sampling strategy: {pair_sampling}"

    center = (np.array(vol_a.shape) - 1.0) / 2.0

    # Generate transform for misalignment simulation
    rot_mat_viz = None
    trans_vec_viz = None
    vol_b_eval = vol_b
    do_transform = rotate_deg > 0 or translate_mm > 0 or affine_scale > 0 or affine_shear_deg > 0
    if do_transform:
        affine_b = case.affines.get(mod_b)
        affine = affine_b if affine_b is not None else case.affines.get(mod_a)
        voxel_sizes_mm = None
        if affine is not None:
            voxel_sizes_mm = np.sqrt(np.sum(np.square(affine[:3, :3]), axis=0))

        if (affine_scale > 0 or affine_shear_deg > 0) and not two_d_only:
            return None, "Affine misalignment currently requires two_d_only=True"

        if affine_scale > 0 or affine_shear_deg > 0:
            lin_mat, trans = generate_random_affine_in_plane_vox(
                rotate_deg=rotate_deg,
                translate_mm=translate_mm,
                scale=affine_scale,
                shear_deg=affine_shear_deg,
                center=center,
                rng=rng,
                voxel_sizes_mm=voxel_sizes_mm,
                snap_translation_to_vox=True,
                fixed_axis=0,
                deterministic=deterministic_transform,
            )
            coords_b = apply_transform_to_coords_vox(coords_a, lin_mat, trans, center)
            vol_b_eval = warp_volume_affine(vol_b, lin_mat, trans, center=center, order=1, cval=0.0)
            rot_mat_viz = lin_mat
            trans_vec_viz = trans
        else:
            gen = generate_random_transform_in_plane_vox if two_d_only else generate_random_transform_vox
            gen_kwargs = {
                "rotate_deg": rotate_deg,
                "translate_mm": translate_mm,
                "center": center,
                "rng": rng,
                "voxel_sizes_mm": voxel_sizes_mm,
                "snap_translation_to_vox": True,
                "deterministic": deterministic_transform,
            }
            if two_d_only:
                gen_kwargs["fixed_axis"] = 0  # axial plane for (z, y, x)
            rot_mat, trans = gen(**gen_kwargs)
            coords_b = apply_transform_to_coords_vox(coords_a, rot_mat, trans, center)
            vol_b_eval = warp_volume_rigid(vol_b, rot_mat, trans, center=center, order=1, cval=0.0)
            rot_mat_viz = rot_mat
            trans_vec_viz = trans
    else:
        coords_b = coords_a.copy()

    # Clip B coordinates to valid range
    shape = np.array(vol_b.shape)
    b_crop_min, b_crop_max = compute_safe_crop_bounds(tuple(int(s) for s in shape), base_margin)
    valid = np.all((coords_b >= b_crop_min) & (coords_b < b_crop_max), axis=1)

    if valid.sum() < 100:
        return None, "Not enough valid coords after transform"

    coords_a = coords_a[valid]
    coords_b = np.round(coords_b[valid]).astype(np.int32)

    # Ensure warped B has signal at the transformed coordinates (reduces wasted extraction).
    abs_b_eval = np.abs(vol_b_eval)
    max_b_eval = float(np.max(abs_b_eval))
    mask_b_eval = None
    if max_b_eval > 0.0:
        mask_b_eval = _infer_signal_mask(vol_b_eval, abs_volume=abs_b_eval, max_abs=max_b_eval)
        mask_b_eval &= abs_b_eval > 0.01 * max_b_eval
        mask_b_eval = safe_preprocess_roi(mask_b_eval, erode_iterations=erode_margin, base_margin=base_margin)
        ok = mask_b_eval[coords_b[:, 0], coords_b[:, 1], coords_b[:, 2]]
        coords_a = coords_a[ok]
        coords_b = coords_b[ok]

    # Extract descriptors
    if pair_sampling == "random_with_kp":
        if not hasattr(extractor, "detect_keypoints_2d") or not hasattr(extractor, "compute_2d"):
            return None, "pair_sampling='random_with_kp' requires a SIFT extractor with 2D keypoint APIs"
        d_a_raw = _extract_with_keypoint_geometry(
            extractor,
            vol_a,
            coords_a,
            mask,
            max_kpts_per_slice=max_kpts_per_slice,
        ).astype(np.float32, copy=False)
        d_b_raw = _extract_with_keypoint_geometry(
            extractor,
            vol_b_eval,
            coords_b,
            roi_mask=mask_b_eval if mask_b_eval is not None else mask,
            max_kpts_per_slice=max_kpts_per_slice,
        ).astype(np.float32, copy=False)
    else:
        d_a_raw = extractor.extract(vol_a, coords_a).astype(np.float32)
        d_b_raw = extractor.extract(vol_b_eval, coords_b).astype(np.float32)

    # Filter valid descriptors
    valid_desc = (np.linalg.norm(d_a_raw, axis=1) > 0.1) & (np.linalg.norm(d_b_raw, axis=1) > 0.1)
    d_a_raw = d_a_raw[valid_desc]
    d_b_raw = d_b_raw[valid_desc]
    coords_a = coords_a[valid_desc]
    coords_b = coords_b[valid_desc]

    if len(d_a_raw) < 32:
        return None, "Too few valid descriptors"

    # Apply PCA (fitted on training data - NO LEAKAGE)
    if pca is not None:
        d_a = pca.transform(d_a_raw).astype(np.float32)
        d_b = pca.transform(d_b_raw).astype(np.float32)
    else:
        d_a = d_a_raw.astype(np.float32)
        d_b = d_b_raw.astype(np.float32)

    return _PreparedCaseData(
        case_id=case_id,
        mod_a=mod_a,
        mod_b=mod_b,
        vol_a=vol_a,
        vol_b=vol_b,
        coords_a=coords_a,
        coords_b=coords_b,
        d_a=d_a,
        d_b=d_b,
        rot_mat_viz=rot_mat_viz,
        trans_vec_viz=trans_vec_viz,
        rotate_deg=rotate_deg,
        translate_mm=translate_mm,
        affine_scale=affine_scale,
        affine_shear_deg=affine_shear_deg,
        two_d_only=two_d_only,
    )


def _compute_matches_and_tre(
    d_a_final: np.ndarray,
    d_b_norm: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    inlier_threshold: float,
    cosine_distances: Optional[np.ndarray] = None,
    ratio_thresh: float = 0.8,
    match_per_slice: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mutual NN matches and TRE values.

    Args:
        d_a_final: Crossed descriptors from A (normalized)
        d_b_norm: Target descriptors from B (normalized)
        coords_a: Coordinates in A
        coords_b: Coordinates in B
        inlier_threshold: TRE threshold for inlier classification
        cosine_distances: Optional precomputed cosine distance matrix.
        ratio_thresh: Lowe's ratio test threshold (1.0 = disabled).
        match_per_slice: If True, restrict matching to candidates with the same
            z-index (`coords[:, 0]`), then merge slice-local matches.

    Returns:
        (idx_a, idx_b, tre_values, is_inlier)
    """
    if match_per_slice:
        idx_a, idx_b, _ = _mutual_nn_matches_same_slice(
            d_a_final,
            d_b_norm,
            coords_a,
            coords_b,
            ratio_thresh=ratio_thresh,
            cosine_distances=cosine_distances,
        )
    else:
        idx_a, idx_b, _ = mutual_nn_matches(
            d_a_final,
            d_b_norm,
            ratio_thresh=ratio_thresh,
            cosine_distances=cosine_distances,
        )
    num_matches = len(idx_a)

    if num_matches > 0:
        matched_coords_b = coords_b[idx_b]
        gt_coords_b = coords_b[idx_a]
        tre_values = np.linalg.norm(matched_coords_b - gt_coords_b, axis=1)
        is_inlier = tre_values < inlier_threshold
    else:
        tre_values = np.array([])
        is_inlier = np.array([], dtype=bool)

    return idx_a, idx_b, tre_values, is_inlier


def _build_result_and_viz(
    prep: _PreparedCaseData,
    d_a_final: np.ndarray,
    d_b_norm: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    tre_values: np.ndarray,
    is_inlier: np.ndarray,
    metrics: Dict[str, float],
    elapsed_time: float,
    return_viz_data: bool,
) -> Union[EvaluationResult, Tuple[EvaluationResult, Optional[VisualizationData]]]:
    """Build EvaluationResult and optional VisualizationData from computed values."""
    # Extract unique z-slices used (in order of first occurrence)
    slices_used = ordered_unique_ints(prep.coords_a[:, 0]) if len(prep.coords_a) > 0 else []

    result = EvaluationResult(
        case_id=prep.case_id,
        top1_acc=metrics["top1_acc"],
        top5_acc=metrics["top5_acc"],
        top10_acc=metrics["top10_acc"],
        median_rank=metrics["median_rank"],
        cosine_sim=metrics["cosine_sim"],
        num_samples=len(prep.d_a),
        num_matches=metrics["num_matches"],
        num_inliers=metrics["num_inliers"],
        inlier_ratio=metrics["inlier_ratio"],
        recall=metrics["recall"],
        spatial_coverage=metrics["spatial_coverage"],
        tre_mean=metrics["tre_mean"],
        tre_median=metrics["tre_median"],
        tre_std=metrics["tre_std"],
        time_s=elapsed_time,
        slices_used=slices_used,
        # SR @ thresholds
        sr_1px=metrics["sr_1px"],
        sr_3px=metrics["sr_3px"],
        sr_5px=metrics["sr_5px"],
        sr_10px=metrics["sr_10px"],
        # AUC @ thresholds
        auc_1px=metrics["auc_1px"],
        auc_3px=metrics["auc_3px"],
        auc_5px=metrics["auc_5px"],
        auc_10px=metrics["auc_10px"],
        tre_values=tre_values.tolist() if len(tre_values) > 0 else [],
    )

    if not return_viz_data:
        return result

    viz_data = VisualizationData(
        case_id=prep.case_id,
        vol_a=prep.vol_a,
        vol_b=prep.vol_b,
        coords_a=prep.coords_a,
        coords_b=prep.coords_b,
        match_idx_a=idx_a,
        match_idx_b=idx_b,
        tre_values=tre_values,
        is_inlier=is_inlier,
        d_a_crossed=d_a_final,
        d_b_norm=d_b_norm,
        modality_a=prep.mod_a,
        modality_b=prep.mod_b,
        top1_acc=metrics["top1_acc"],
        inlier_ratio=metrics["inlier_ratio"],
        tre_mean=metrics["tre_mean"],
        tre_median=metrics["tre_median"],
        rotate_deg=prep.rotate_deg,
        translate_mm=prep.translate_mm,
        affine_scale=prep.affine_scale,
        affine_shear_deg=prep.affine_shear_deg,
        rot_mat=prep.rot_mat_viz,
        trans_vec=prep.trans_vec_viz,
        two_d_only=prep.two_d_only,
    )
    return result, viz_data


def _resolve_vae_architecture_flags(
    config: Dict[str, Any],
    checkpoint: Dict[str, Any],
    state_dict: Dict[str, Any],
) -> Tuple[int, bool, bool]:
    """Resolve VAE architecture flags from checkpoint state_dict.

    Inspects actual checkpoint keys to determine variational architecture,
    falling back to config values when checkpoint keys are absent.
    """
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    fsq_levels = config.get("fsq_levels", checkpoint_config.get("fsq_levels", 0))
    fsq_levels = int(fsq_levels) if fsq_levels is not None else 0

    variational = bool(config.get("variational", checkpoint_config.get("variational", False)))
    variational_app_only = bool(
        config.get("variational_app_only", checkpoint_config.get("variational_app_only", False))
    )

    has_logvar_app = any(k.startswith("logvar_app_head.") for k in state_dict.keys())
    has_logvar_geom = any(k.startswith("logvar_geom_head.") for k in state_dict.keys())

    if has_logvar_app or has_logvar_geom:
        variational = True
        variational_app_only = has_logvar_app and not has_logvar_geom

    return fsq_levels, variational, variational_app_only


def _remap_legacy_vae_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle legacy DisentangledVAECrosser checkpoints from before Dropout/Identity
    layers were inserted into `app_crosser.film_net` and `app_crosser.transform`.
    """
    remapped: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        if "app_crosser.film_net.2." in key:
            new_key = key.replace("app_crosser.film_net.2.", "app_crosser.film_net.3.")
        elif "app_crosser.transform.2." in key:
            new_key = key.replace("app_crosser.transform.2.", "app_crosser.transform.3.")
        remapped[new_key] = value
    return remapped


def mutual_nn_matches(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    ratio_thresh: float = 0.8,
    cosine_distances: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find mutual nearest neighbor matches with optional ratio test.

    Args:
        desc_a: Descriptors from modality A (N, D)
        desc_b: Descriptors from modality B (M, D)
        ratio_thresh: Lowe's ratio test threshold (1.0 = disabled).
        cosine_distances: Optional precomputed cosine distance matrix (N, M).
            If provided, avoids recomputing `cdist`.

    Returns:
        idx_a: Indices in desc_a
        idx_b: Indices in desc_b
        distances: Match cosine distances
    """
    if len(desc_a) == 0 or len(desc_b) == 0:
        return np.array([]), np.array([]), np.array([])

    if cosine_distances is not None:
        distances = cosine_distances
    else:
        distances = cdist(desc_a, desc_b, metric="cosine")

    forward = np.argmin(distances, axis=1)
    backward = np.argmin(distances, axis=0)
    mutual = backward[forward] == np.arange(len(desc_a))

    if ratio_thresh < 1.0 and distances.shape[1] >= 2:
        sorted_dists = np.sort(distances, axis=1)
        ratios = sorted_dists[:, 0] / (sorted_dists[:, 1] + 1e-8)
        mutual = mutual & (ratios < ratio_thresh)

    idx_a = np.where(mutual)[0]
    idx_b = forward[idx_a]
    match_distances = distances[idx_a, idx_b]

    return idx_a, idx_b, match_distances


def _mutual_nn_matches_same_slice(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    ratio_thresh: float = 0.8,
    cosine_distances: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run mutual-NN matching independently per z-slice and concatenate results."""
    if len(desc_a) == 0 or len(desc_b) == 0:
        return np.array([]), np.array([]), np.array([])

    if coords_a.shape[0] != len(desc_a):
        raise ValueError(
            f"coords_a rows must match desc_a rows: {coords_a.shape[0]} != {len(desc_a)}"
        )
    if coords_b.shape[0] != len(desc_b):
        raise ValueError(
            f"coords_b rows must match desc_b rows: {coords_b.shape[0]} != {len(desc_b)}"
        )

    z_a = np.asarray(coords_a[:, 0], dtype=np.int64)
    z_b = np.asarray(coords_b[:, 0], dtype=np.int64)
    common_z = np.intersect1d(np.unique(z_a), np.unique(z_b))
    if common_z.size == 0:
        return np.array([]), np.array([]), np.array([])

    idx_a_parts: list[np.ndarray] = []
    idx_b_parts: list[np.ndarray] = []
    dist_parts: list[np.ndarray] = []

    for z in common_z.tolist():
        rows = np.where(z_a == int(z))[0]
        cols = np.where(z_b == int(z))[0]
        if rows.size == 0 or cols.size == 0:
            continue

        local_cosine = cosine_distances[np.ix_(rows, cols)] if cosine_distances is not None else None

        local_idx_a, local_idx_b, local_dists = mutual_nn_matches(
            desc_a[rows],
            desc_b[cols],
            ratio_thresh=ratio_thresh,
            cosine_distances=local_cosine,
        )
        if len(local_idx_a) == 0:
            continue

        idx_a_parts.append(rows[local_idx_a].astype(np.int64, copy=False))
        idx_b_parts.append(cols[local_idx_b].astype(np.int64, copy=False))
        dist_parts.append(local_dists.astype(np.float64, copy=False))

    if not idx_a_parts:
        return np.array([]), np.array([]), np.array([])

    idx_a = np.concatenate(idx_a_parts, axis=0)
    idx_b = np.concatenate(idx_b_parts, axis=0)
    match_distances = np.concatenate(dist_parts, axis=0)

    # Keep output order deterministic and stable by query index.
    order = np.argsort(idx_a, kind="stable")
    return idx_a[order], idx_b[order], match_distances[order]


def compute_retrieval_metrics(
    crossed_a: np.ndarray,
    targets_b: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    inlier_threshold: float = 5.0,
    similarities: Optional[np.ndarray] = None,
    cosine_distances: Optional[np.ndarray] = None,
    normalized_inputs: bool = False,
    ratio_thresh: float = 0.8,
    match_per_slice: bool = False,
) -> Dict[str, float]:
    """
    Compute comprehensive retrieval and matching metrics.

    Args:
        crossed_a: Crossed descriptors from modality A
        targets_b: Target descriptors from modality B
        coords_a: Coordinates in A (original space)
        coords_b: Coordinates in B (may be transformed for misaligned evaluation)
        inlier_threshold: Distance threshold for inlier classification
        similarities: Optional precomputed cosine similarity matrix.
        cosine_distances: Optional precomputed cosine distance matrix.
        normalized_inputs: If True, `crossed_a` and `targets_b` are already L2-normalized.
        match_per_slice: If True, restricts matching to candidates from the same
            z-slice (`coords[:, 0]`). Retrieval ranking metrics remain global.

    Note:
        coords_a[i] and coords_b[i] are corresponding points (same index = ground truth match).
        For aligned data: coords_b == coords_a. For misaligned: coords_b = transform(coords_a).

    Returns dict with:
        - top1_acc, top5_acc, top10_acc: Retrieval accuracy at k
        - median_rank, cosine_sim: Retrieval quality
        - num_matches, num_inliers: Absolute counts
        - inlier_ratio: Precision = inliers / matches
        - recall: Inliers / total samples (fraction of GT pairs correctly matched)
        - spatial_coverage: Normalized spread of inlier coords (0-1, higher = better)
        - tre_mean, tre_median, tre_std: Target Registration Error stats
    """
    n = len(crossed_a)
    if n == 0:
        return {
            "top1_acc": 0.0, "top5_acc": 0.0, "top10_acc": 0.0,
            "median_rank": float("inf"), "cosine_sim": 0.0,
            "num_matches": 0, "num_inliers": 0, "inlier_ratio": 0.0,
            "recall": 0.0, "spatial_coverage": 0.0,
            "tre_mean": float("inf"), "tre_median": float("inf"), "tre_std": 0.0,
        }

    # Normalize before descriptor-space comparisons. Retrieval is scale-invariant.
    if normalized_inputs:
        crossed_a_norm = crossed_a
        targets_b_norm = targets_b
    else:
        crossed_a_norm = crossed_a / (np.linalg.norm(crossed_a, axis=1, keepdims=True) + 1e-8)
        targets_b_norm = targets_b / (np.linalg.norm(targets_b, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarities (higher is better)
    if similarities is None:
        similarities = crossed_a_norm @ targets_b_norm.T

    ranking_scores = similarities

    # Retrieval metrics (ground truth is same index for aligned data).
    # Vectorized inverse-rank computation avoids Python loops.
    m = ranking_scores.shape[1]
    gt = np.arange(n, dtype=np.int32)
    ranks = np.full(n, fill_value=max(1, m), dtype=np.int32)
    valid_gt = gt < m
    if np.any(valid_gt):
        scores_valid = ranking_scores[valid_gt]
        order = np.argsort(-scores_valid, axis=1)
        inv_order = np.empty_like(order, dtype=np.int32)
        row_idx = np.arange(order.shape[0])[:, None]
        inv_order[row_idx, order] = np.arange(m, dtype=np.int32)
        gt_valid = gt[valid_gt]
        ranks[valid_gt] = inv_order[np.arange(len(gt_valid)), gt_valid] + 1

    cosines = np.zeros(n, dtype=np.float32)
    diag_n = min(n, similarities.shape[1])
    if diag_n > 0:
        diag_idx = np.arange(diag_n)
        cosines[:diag_n] = similarities[diag_idx, diag_idx].astype(np.float32, copy=False)

    # Matching metrics using mutual NN (pass pre-computed Sinkhorn scores)
    if cosine_distances is None:
        cosine_distances = 1.0 - similarities
    if match_per_slice:
        idx_a, idx_b, match_dists = _mutual_nn_matches_same_slice(
            crossed_a_norm,
            targets_b_norm,
            coords_a,
            coords_b,
            ratio_thresh=ratio_thresh,
            cosine_distances=cosine_distances,
        )
    else:
        idx_a, idx_b, match_dists = mutual_nn_matches(
            crossed_a_norm,
            targets_b_norm,
            ratio_thresh=ratio_thresh,
            cosine_distances=cosine_distances,
        )
    num_matches = len(idx_a)

    # TRE computation (works with any number of matches)
    spatial_coverage = 0.0
    # Standard thresholds for SR/AUC metrics (in pixels for 2D matching)
    sr_auc_thresholds = [1.0, 3.0, 5.0, 10.0]

    if num_matches > 0:
        matched_coords_b = coords_b[idx_b]  # Where matches were found
        gt_coords_b = coords_b[idx_a]  # Ground truth: where idx_a points should be in B

        # Compute TRE between matched coords and ground truth
        tre_values = np.linalg.norm(matched_coords_b - gt_coords_b, axis=1)
        inlier_mask = tre_values < inlier_threshold
        num_inliers = int(inlier_mask.sum())

        tre_mean = float(np.mean(tre_values))
        tre_median = float(np.median(tre_values))
        tre_std = float(np.std(tre_values))

        # Compute SR @ thresholds and AUC @ thresholds
        sr_values, auc_values = compute_sr_auc_from_tre(tre_values, sr_auc_thresholds)

        # Compute spatial coverage of inliers
        # Coverage = mean of normalized std in each dimension (0-1 scale)
        # Higher = inliers are more spread out across the volume
        if num_inliers >= 3:
            inlier_coords = coords_a[idx_a[inlier_mask]]
            # Compute range of all sampled coords as normalization
            coord_range = coords_a.max(axis=0) - coords_a.min(axis=0) + 1e-8
            # Normalized std: std / range for each dimension
            norm_std = np.std(inlier_coords, axis=0) / coord_range
            spatial_coverage = float(np.mean(norm_std))
    else:
        num_inliers = 0
        tre_mean = float("inf")
        tre_median = float("inf")
        tre_std = 0.0
        sr_values = [0.0] * len(sr_auc_thresholds)
        auc_values = [0.0] * len(sr_auc_thresholds)

    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
    recall = num_inliers / n  # Fraction of GT pairs correctly matched

    return {
        "top1_acc": float((ranks == 1).mean()),
        "top5_acc": float((ranks <= 5).mean()),
        "top10_acc": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)),
        "cosine_sim": float(np.mean(cosines)),
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "recall": recall,
        "spatial_coverage": spatial_coverage,
        "tre_mean": tre_mean,
        "tre_median": tre_median,
        "tre_std": tre_std,
        # SR @ thresholds (Success Rate: binary per-case metric)
        "sr_1px": sr_values[0],
        "sr_3px": sr_values[1],
        "sr_5px": sr_values[2],
        "sr_10px": sr_values[3],
        # AUC @ thresholds (Area Under Curve: continuous per-case metric)
        "auc_1px": auc_values[0],
        "auc_3px": auc_values[1],
        "auc_5px": auc_values[2],
        "auc_10px": auc_values[3],
    }


def evaluate_case(
    case: CaseData,
    modalities: List[str],
    extractor,
    pca,
    model,
    device: str,
    rotate_deg: float = 0.0,
    translate_mm: float = 0.0,
    affine_scale: float = 0.0,
    affine_shear_deg: float = 0.0,
    deterministic_transform: bool = False,
    n_samples: int = 2000,
    inlier_threshold: float = 5.0,
    seed: int = 42,
    adapter: Optional[BaseAdapter] = None,
    return_viz_data: bool = False,
    two_d_only: bool = True,
    pair_sampling: str = "random",
    max_kpts_per_slice: int = 500,
    z_slices: Optional[List[int]] = None,
    mod_a_idx: int = 0,
    mod_b_idx: int = 1,
    ratio_thresh: float = 0.8,
    match_per_slice: bool = False,
) -> Union[EvaluationResult, Tuple[EvaluationResult, Optional[VisualizationData]]]:
    """
    Evaluate on a single case with optional misalignment simulation.

    NO DATA LEAKAGE: Uses PCA fitted on training data.

    Args:
        two_d_only: If True (default), constrains transforms to axial in-plane only.
            This is required for 2D descriptors like SIFT which cannot handle
            out-of-plane rotations.
        ratio_thresh: Lowe's ratio test threshold (1.0 = disabled).
        match_per_slice: If True, restricts evaluation matching to same-z candidates.
            Useful for slice-faithful medical visualizations.
    """
    t0 = time.time()

    # Prepare case data (sampling, extraction, PCA)
    prep_result = _prepare_case_data(
        case, modalities, extractor, pca,
        rotate_deg, translate_mm, affine_scale, affine_shear_deg, n_samples, seed,
        two_d_only, pair_sampling, deterministic_transform, max_kpts_per_slice,
        z_slices=z_slices
    )

    # Handle preparation errors
    if not isinstance(prep_result, _PreparedCaseData):
        _, error_msg = prep_result
        error_result = _make_error_result(case.case_id, error_msg, time.time() - t0)
        if return_viz_data:
            return error_result, None
        return error_result

    prep = prep_result

    # Apply crossing model
    with torch.no_grad():
        d_a_t = torch.from_numpy(prep.d_a).to(device=device, dtype=torch.float32)
        mod_a_tensor = torch.full((len(prep.d_a),), mod_a_idx, dtype=torch.long, device=device)
        mod_b_tensor = torch.full((len(prep.d_a),), mod_b_idx, dtype=torch.long, device=device)
        d_a_crossed = model(d_a_t, mod_a_tensor, mod_b_tensor, normalize=True).cpu().numpy()

    # Normalize targets (before adapter - BaseAdapter interface requires L2-normalized inputs)
    d_b_norm = prep.d_b / (np.linalg.norm(prep.d_b, axis=1, keepdims=True) + 1e-8)

    # Apply test-time adapter (if provided)
    if adapter is not None:
        d_a_crossed = adapter.adapt(d_a_crossed, d_b_norm, prep.coords_a, prep.coords_b)

    # Shared normalized descriptors + pairwise matrices (used by matching + retrieval)
    d_a_crossed_norm = d_a_crossed / (np.linalg.norm(d_a_crossed, axis=1, keepdims=True) + 1e-8)
    d_b_norm_unit = d_b_norm / (np.linalg.norm(d_b_norm, axis=1, keepdims=True) + 1e-8)
    sim_matrix = d_a_crossed_norm @ d_b_norm_unit.T
    cosine_distances = 1.0 - sim_matrix

    # Compute matches and TRE
    idx_a, idx_b, tre_values, is_inlier = _compute_matches_and_tre(
        d_a_crossed_norm, d_b_norm_unit, prep.coords_a, prep.coords_b, inlier_threshold,
        cosine_distances=cosine_distances,
        ratio_thresh=ratio_thresh,
        match_per_slice=match_per_slice,
    )

    # Compute full metrics (retrieval metrics stay unaffected by filtering)
    metrics = compute_retrieval_metrics(
        d_a_crossed_norm, d_b_norm_unit, prep.coords_a, prep.coords_b, inlier_threshold,
        similarities=sim_matrix,
        cosine_distances=cosine_distances,
        normalized_inputs=True,
        ratio_thresh=ratio_thresh,
        match_per_slice=match_per_slice,
    )

    return _build_result_and_viz(
        prep, d_a_crossed, d_b_norm, idx_a, idx_b, tre_values, is_inlier,
        metrics, time.time() - t0, return_viz_data
    )


def evaluate_case_no_crossing(
    case: CaseData,
    modalities: List[str],
    extractor,
    pca,
    device: str,
    rotate_deg: float = 0.0,
    translate_mm: float = 0.0,
    affine_scale: float = 0.0,
    affine_shear_deg: float = 0.0,
    deterministic_transform: bool = False,
    n_samples: int = 2000,
    inlier_threshold: float = 5.0,
    seed: int = 42,
    two_d_only: bool = True,
    pair_sampling: str = "random",
    max_kpts_per_slice: int = 500,
    z_slices: Optional[List[int]] = None,
    ratio_thresh: float = 0.8,
    match_per_slice: bool = False,
) -> Tuple[EvaluationResult, Optional[VisualizationData]]:
    """
    Evaluate without crossing model (just normalized PCA descriptors).

    Used for crossing comparison visualization.

    Args:
        two_d_only: If True (default), constrains transforms to axial in-plane only.
        ratio_thresh: Lowe's ratio test threshold (1.0 = disabled).
        match_per_slice: If True, restricts evaluation matching to same-z candidates.
    """
    t0 = time.time()

    # Prepare case data (sampling, extraction, PCA)
    prep_result = _prepare_case_data(
        case, modalities, extractor, pca,
        rotate_deg, translate_mm, affine_scale, affine_shear_deg, n_samples, seed,
        two_d_only, pair_sampling, deterministic_transform, max_kpts_per_slice,
        z_slices=z_slices
    )

    # Handle preparation errors
    if not isinstance(prep_result, _PreparedCaseData):
        _, error_msg = prep_result
        error_result = _make_error_result(case.case_id, error_msg, time.time() - t0)
        return error_result, None

    prep = prep_result

    # NO CROSSING - just normalize descriptors
    d_a_norm = prep.d_a / (np.linalg.norm(prep.d_a, axis=1, keepdims=True) + 1e-8)
    d_b_norm = prep.d_b / (np.linalg.norm(prep.d_b, axis=1, keepdims=True) + 1e-8)
    sim_matrix = d_a_norm @ d_b_norm.T
    cosine_distances = 1.0 - sim_matrix

    # Compute matches and TRE
    idx_a, idx_b, tre_values, is_inlier = _compute_matches_and_tre(
        d_a_norm,
        d_b_norm,
        prep.coords_a,
        prep.coords_b,
        inlier_threshold,
        cosine_distances=cosine_distances,
        ratio_thresh=ratio_thresh,
        match_per_slice=match_per_slice,
    )

    # Compute full metrics
    metrics = compute_retrieval_metrics(
        d_a_norm,
        d_b_norm,
        prep.coords_a,
        prep.coords_b,
        inlier_threshold,
        similarities=sim_matrix,
        cosine_distances=cosine_distances,
        normalized_inputs=True,
        ratio_thresh=ratio_thresh,
        match_per_slice=match_per_slice,
    )

    return _build_result_and_viz(
        prep, d_a_norm, d_b_norm, idx_a, idx_b, tre_values, is_inlier,
        metrics, time.time() - t0, return_viz_data=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CrossFeat model with proper train/test separation"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing best_model.pt, pca.pkl, config.json")
    parser.add_argument("--model", type=str, default="best_model",
                        help="Name of the model used for evaluation.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., remind, resect, deliver, eventscape, brats, whu_opt_sar, qxs_saropt). "
                             "Auto-detected from model config if not specified.")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Override default data root for the dataset")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Legacy: data directory (deprecated, use --dataset_root instead)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--use_full", action="store_true")
    parser.add_argument("--modalities", nargs=2, default=None, metavar=("MOD_A", "MOD_B"),
                        help="Modality pair to evaluate (e.g., --modalities t2 us). "
                             "For universal models trained on multiple pairs. "
                             "If not specified, uses modalities from config.")

    # Augmentation (on-the-fly for testing generalization)
    parser.add_argument("--rotate_deg", type=float, default=0.0,
                        help="Rotation for misalignment test (0 = aligned)")
    parser.add_argument("--translate_mm", type=float, default=0.0,
                        help="Translation for misalignment test (0 = aligned)")
    parser.add_argument(
        "--deterministic_transform",
        action="store_true",
        help="If set, apply the exact requested transform parameters instead of sampling randomly "
             "from ranges. Useful for sweep curves and sanity checks (e.g., 360° == identity).",
    )
    parser.add_argument(
        "--affine_scale",
        type=float,
        default=0.0,
        help="Max fractional scale for affine misalignment (0 disables). "
             "Interpreted as per-axis scale sampled from [1-affine_scale, 1+affine_scale]. "
             "Requires two_d_only=True.",
    )
    parser.add_argument(
        "--affine_shear_deg",
        type=float,
        default=0.0,
        help="Max shear angle (deg) for affine misalignment (0 disables). Requires two_d_only=True.",
    )
    parser.add_argument(
        "--allow_3d_rotation",
        action="store_true",
        help="Allow 3D (out-of-plane) rotation. WARNING: Not recommended for 2D descriptors like SIFT. "
             "By default, only axial in-plane transforms are applied (two_d_only=True).",
    )

    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument(
        "--max_kpts_per_slice",
        type=int,
        default=None,
        help="Override the model config's max_kpts_per_slice (keypoints per axial slice). "
        "Default: use model config value (typically 500). Set high (e.g. 10000) for uncapped evaluation.",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=1,
        help="Worker threads for per-case evaluation (only used when no viz/adapter).",
    )
    parser.add_argument(
        "--baseline_workers",
        type=int,
        default=0,
        help="Max concurrent baseline evaluations (0 = auto: max(1, eval_workers // 4)).",
    )
    parser.add_argument("--inlier_threshold", type=float, default=5.0,
                        help="Distance threshold for inlier classification (voxels)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pair_sampling",
        type=str,
        default=None,
        choices=["random", "random_with_kp", "sift_union", "2d_sift_ref", "2d_sift_dual"],
        help="Override pair_sampling from the saved config.json (optional).",
    )
    parser.add_argument(
        "--slice_sampling",
        type=str,
        default="sift",
        choices=["sift", "uniform_random"],
        help="Slice selection strategy: 'sift' (default, ranked by ROI area) or "
             "'uniform_random' (uniformly random selection for fair baseline comparison).",
    )
    parser.add_argument(
        "--num_slices",
        type=int,
        default=None,
        help="Number of slices to sample. For uniform_random: selects N random slices. "
             "For sift: selects top-N slices by ROI area. "
             "Default: None (use all valid slices).",
    )
    parser.add_argument(
        "--match_per_slice",
        action="store_true",
        help="Restrict matching to same-z candidates only (slice-local matching). "
             "Intended for medical slice-faithful visualization/analysis.",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (optional)")

    # Test-time adaptation
    parser.add_argument("--adapter", type=str, default=None,
                        choices=["none", "inlier_ridge"],
                        help="Test-time adaptation method (default: none)")
    parser.add_argument("--adapter_n_iterations", type=int, default=1,
                        help="Number of iterations for inlier_ridge adapter")
    parser.add_argument("--adapter_ridge_alpha", type=float, default=1.0,
                        help="Ridge regularization for inlier_ridge adapter")

    # Baselines
    add_baseline_args(parser)

    parser.add_argument(
        "--ratio_thresh",
        type=float,
        default=0.8,
        help="Lowe's ratio test threshold for mutual-NN matching. "
        "Lower = stricter (more matches rejected). 1.0 = disabled.",
    )

    # Visualization flags
    parser.add_argument("--viz", action="store_true",
                        help="Enable visualization: multi-view matches and TRE distribution")
    parser.add_argument("--viz_compare_cross", action="store_true",
                        help="Compare crossing vs. no crossing (side-by-side figure)")
    parser.add_argument("--viz_compare_tta", action="store_true",
                        help="Compare TTA vs. no TTA (requires --adapter flag)")
    parser.add_argument("--zscore", action="store_true",
                        help="Apply volume-level z-score normalization after loading "
                             "(for models trained with z-score preprocessing)")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="Maximum number of cases to evaluate (default: all)")
    parser.add_argument("--viz_max_cases", type=int, default=5,
                        help="Maximum number of cases to visualize (default: 5)")

    args = parser.parse_args()

    # Set two_d_only based on --allow_3d_rotation flag
    # Default is True (only axial in-plane transforms) for safety with 2D descriptors
    args.two_d_only = not args.allow_3d_rotation

    model_dir = Path(args.model_dir)

    # Load config with error handling
    config_path = model_dir / "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)

    print("=" * 120)
    print("CrossFeat Evaluation (No Data Leakage)")
    print("=" * 120)
    print(f"Model dir: {model_dir}")
    print(f"Descriptor: {config['descriptor']}")
    print(f"Model: {config['model']}")
    print(f"Training aug: rotate={config.get('aug_rotate_deg', 0)}°, translate={config.get('aug_translate_mm', 0)}mm")
    eval_affine_str = ""
    if args.affine_scale > 0 or args.affine_shear_deg > 0:
        eval_affine_str = f", scale±{args.affine_scale:.2f}, shear±{args.affine_shear_deg:.1f}°"
    if args.deterministic_transform:
        print(f"Eval aug (deterministic): rotate={args.rotate_deg}°, translate={args.translate_mm}mm{eval_affine_str}")
    else:
        print(f"Eval aug (max): rotate≤{args.rotate_deg}°, translate≤{args.translate_mm}mm{eval_affine_str}")
    pair_sampling = str(config.get("pair_sampling", "random"))
    if args.pair_sampling is not None:
        pair_sampling = str(args.pair_sampling)
    print(f"Pair sampling: {pair_sampling}")
    slice_sampling_str = args.slice_sampling
    if args.num_slices is not None:
        slice_sampling_str = f"{args.slice_sampling} (n={args.num_slices})"
    print(f"Slice sampling: {slice_sampling_str}")
    print(f"Split: {args.split}")
    print()

    # Load PCA (if the model was trained with PCA)
    if config.get("use_pca", True):
        pca_path = model_dir / "pca.pkl"
        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: PCA file not found: {pca_path}")
            sys.exit(1)
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error: Corrupted PCA file: {e}")
            sys.exit(1)
        print(f"PCA: {pca.n_components_} dims (fitted on training data)")
    else:
        pca = None
        print("PCA: disabled (model trained on raw descriptors)")

    # Determine modalities and their indices
    # For universal models (3+ modalities), --modalities specifies which pair to evaluate
    modality_to_idx = config.get("modality_to_idx", {})
    canonical_map_raw = config.get("canonical_map", {})
    is_multi_dataset = config.get("is_multi_dataset", False)
    if not is_multi_dataset and "is_multi_dataset" not in config and "datasets" in config:
        # Fallback: infer multi-dataset from presence of "datasets" key
        # only when is_multi_dataset was not explicitly set
        is_multi_dataset = True

    # For multi-dataset configs, build a flat modalities list from modality_to_idx
    # and resolve dataset-qualified keys (e.g. "deliver.img") from --dataset + --modalities
    if is_multi_dataset:
        all_modalities = list(modality_to_idx.keys())  # e.g. ["remind.t2", ..., "deliver.img"]
        # Extract bare modality names per dataset for validation
        dataset_modalities: dict[str, list[str]] = {}
        for ds_entry in config.get("datasets", []):
            dataset_modalities[ds_entry["name"]] = ds_entry["modalities"]
    else:
        all_modalities = config.get("modalities", [])

    if args.modalities:
        # User specified a modality pair (for universal models)
        mod_a, mod_b = args.modalities

        if is_multi_dataset:
            # Resolve dataset name for namespaced lookup.
            # For cross-dataset eval (e.g., --dataset resect), the model was
            # trained on a different dataset (e.g., remind).  We need the
            # *training* dataset namespace for modality index lookup, not the
            # eval dataset.  Fall back to the first training dataset whose
            # modalities contain the requested pair.
            eval_dataset = args.dataset or config.get("dataset", "remind")
            model_namespace = eval_dataset  # default: same as eval dataset

            def _mod_in_ds(mod: str, ds_mod_list: list[str]) -> bool:
                """Check if modality (or any equivalent name) is in a dataset's modality list."""
                ds_set = set(ds_mod_list)
                return bool(get_equivalents(mod) & ds_set)

            ds_mods = dataset_modalities.get(eval_dataset, [])
            if not ds_mods or not all(_mod_in_ds(mod, ds_mods) for mod in (mod_a, mod_b)):
                # eval_dataset not in training config — find a compatible one
                for ds_name, ds_mod_list in dataset_modalities.items():
                    if all(_mod_in_ds(m, ds_mod_list) for m in (mod_a, mod_b)):
                        model_namespace = ds_name
                        ds_mods = ds_mod_list
                        print(f"  Cross-dataset eval: using '{model_namespace}' "
                              f"namespace for modality indices (eval data: '{eval_dataset}')")
                        break
                else:
                    # Last resort: check if modalities exist directly in modality_to_idx
                    # (canonical/universal models store bare names like "rgb", "thermal")
                    if all(
                        find_modality_index(m, modality_to_idx) is not None
                        for m in (mod_a, mod_b)
                    ):
                        model_namespace = None  # skip namespaced lookup
                        print(f"  Cross-dataset eval: modalities found directly in "
                              f"modality_to_idx (no namespace needed)")
                    else:
                        print(f"Error: modalities {[mod_a, mod_b]} not found in any training dataset")
                        print(f"Available datasets: {list(dataset_modalities.keys())}")
                        print(f"modality_to_idx: {modality_to_idx}")
                        sys.exit(1)

            # Validate modalities exist in the resolved namespace (with alias fallback)
            if model_namespace is not None:
                for mod in (mod_a, mod_b):
                    if not _mod_in_ds(mod, ds_mods):
                        print(f"Error: modality '{mod}' not in dataset '{model_namespace}' modalities: {ds_mods}")
                        print(f"Available datasets: {list(dataset_modalities.keys())}")
                        sys.exit(1)

            # Look up indices using namespaced keys, with canonical map + alias fallback
            for mod_name, label in [(mod_a, "source"), (mod_b, "target")]:
                # When model_namespace is None, skip namespaced lookup
                # and go straight to direct modality_to_idx lookup
                if model_namespace is None:
                    idx = find_modality_index(mod_name, modality_to_idx)
                    if idx is not None:
                        print(f"  Resolved {mod_name} directly from modality_to_idx (index {idx})")
                        if label == "source":
                            mod_a_idx = idx
                        else:
                            mod_b_idx = idx
                        continue
                    print(f"Error: could not find index for {label} modality '{mod_name}' "
                          f"in modality_to_idx: {modality_to_idx}")
                    sys.exit(1)

                ns_key = f"{model_namespace}.{mod_name}"
                # Try canonical_map first (handles canonical modality merging)
                canon_key = canonical_map_raw.get(ns_key)
                if canon_key is not None:
                    idx = modality_to_idx.get(canon_key)
                    if idx is not None:
                        print(f"  Resolved {mod_name} via canonical mapping: "
                              f"{ns_key} -> {canon_key} (index {idx})")
                else:
                    idx = None

                # Fallback: direct namespaced lookup (non-canonical models)
                if idx is None:
                    idx = find_modality_index(ns_key, modality_to_idx)

                # Fallback: alias resolution
                if idx is None:
                    alias = resolve_modality_alias(mod_name)
                    alias_key = f"{model_namespace}.{alias}"
                    # Try canonical_map for aliased name
                    canon_alias = canonical_map_raw.get(alias_key)
                    if canon_alias is not None:
                        idx = modality_to_idx.get(canon_alias)
                    if idx is None:
                        idx = modality_to_idx.get(alias_key)
                    if idx is not None:
                        print(f"  Note: mapping {mod_name} -> {alias} (alias) for {label} modality")

                # Final fallback: bare modality name (for models with un-namespaced keys)
                if idx is None:
                    idx = find_modality_index(mod_name, modality_to_idx)
                    if idx is not None:
                        print(f"  Resolved {mod_name} via bare key fallback (index {idx})")

                if idx is None:
                    print(f"Error: could not find index for {label} modality '{mod_name}' "
                          f"(tried '{ns_key}', canonical_map, aliases, and bare key) in modality_to_idx: "
                          f"{list(modality_to_idx.keys())}")
                    sys.exit(1)
                if label == "source":
                    mod_a_idx = idx
                else:
                    mod_b_idx = idx
        else:
            # Standard single-dataset model: validate with alias fallback
            base_modalities_set = set(all_modalities)
            for mod in (mod_a, mod_b):
                if mod not in base_modalities_set:
                    alias = resolve_modality_alias(mod)
                    if alias != mod and alias in base_modalities_set:
                        print(f"  Note: mapping {mod} -> {alias} (alias)")
                    elif alias not in base_modalities_set:
                        print(f"Error: modality '{mod}' not in model's modalities: {all_modalities}")
                        sys.exit(1)

            # Get modality indices for model forward pass (alias-aware)
            if modality_to_idx:
                mod_a_idx_result = find_modality_index(mod_a, modality_to_idx)
                mod_b_idx_result = find_modality_index(mod_b, modality_to_idx)
                if mod_a_idx_result is None:
                    print(f"Error: could not resolve index for '{mod_a}' in {list(modality_to_idx.keys())}")
                    sys.exit(1)
                if mod_b_idx_result is None:
                    print(f"Error: could not resolve index for '{mod_b}' in {list(modality_to_idx.keys())}")
                    sys.exit(1)
                mod_a_idx = mod_a_idx_result
                mod_b_idx = mod_b_idx_result
            else:
                # Fallback: assume order from config["modalities"], with alias
                try:
                    mod_a_idx = all_modalities.index(mod_a)
                except ValueError:
                    mod_a_idx = all_modalities.index(resolve_modality_alias(mod_a))
                try:
                    mod_b_idx = all_modalities.index(mod_b)
                except ValueError:
                    mod_b_idx = all_modalities.index(resolve_modality_alias(mod_b))

        modalities = [mod_a, mod_b]
        print(f"Evaluating pair: {mod_a} -> {mod_b} (indices: {mod_a_idx} -> {mod_b_idx})")
    else:
        if is_multi_dataset:
            # Multi-dataset models always require explicit --modalities
            print(f"Error: This is a multi-dataset universal model with {len(all_modalities)} modalities.")
            print("You must specify which pair to evaluate with --dataset NAME --modalities MOD_A MOD_B")
            print("Available datasets and modalities:")
            for ds_name, ds_mods in dataset_modalities.items():
                print(f"  {ds_name}: {ds_mods}")
            sys.exit(1)
        else:
            # Default behavior: use modalities from config (standard 2-modality model)
            base_modalities = all_modalities

            # Universal models (3+ modalities) require explicit --modalities argument
            if len(base_modalities) > 2:
                print(f"Error: This is a universal model trained on {len(base_modalities)} modalities: {base_modalities}")
                print("You must specify which pair to evaluate with --modalities MOD_A MOD_B")
                print("Example: --modalities t2 us")
                sys.exit(1)

            # Always use clean modality names (no _full suffix); manifest handles that
            modalities = list(base_modalities)

            # Default indices (0 and 1) for standard 2-modality models
            mod_a_idx, mod_b_idx = 0, 1

    pca_dim = config["pca_dim"]
    model_descriptor_dim = pca_dim if pca is not None else config.get("raw_descriptor_dim", pca_dim)

    model_name = str(config.get("model", "vae")).lower()
    if model_name != "vae":
        print(
            f"Error: model={model_name!r} is unsupported in CrossFeat. "
            "Only model='vae' is supported."
        )
        sys.exit(1)

    checkpoint = torch.load(model_dir / f"{args.model}.pt", map_location=args.device, weights_only=False)

    # Resolve VAE architecture flags from checkpoint topology (not just config.json)
    state_dict = checkpoint.get("model_state_dict", {})
    _vae_fsq, _vae_var, _vae_var_app = _resolve_vae_architecture_flags(
        config, checkpoint, state_dict
    )

    # Use num_modalities from config (not evaluation pair) to match trained model shape
    num_modalities = len(all_modalities)
    model = DisentangledVAECrosser(
        descriptor_dim=model_descriptor_dim,
        num_modalities=num_modalities,
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.0),
        d_geom=config.get("d_geom", 64),
        d_app=config.get("d_app", 64),
        embed_dim=config.get("embed_dim", 16),
        simple_crosser=config.get("simple_crosser", False),
        separate_encoder=config.get("separate_encoder", False),
        modality_specific_encoder=config.get("modality_specific_encoder", False),
        normalize_geom=config.get("normalize_geom", False),
        geom_scale=config.get("geom_scale", 1.0),
        use_adversarial=config.get("adversarial", False),
        condition_decoder=config.get("condition_decoder", False),
        fsq_levels=_vae_fsq,
        variational=_vae_var,
        variational_app_only=_vae_var_app,
        use_grad_head=config.get("vae_grad_head", 0.0) > 0,
        use_modality_heads=config.get("vae_modality_heads", 0.0) > 0,
        use_coord_head=config.get("vae_coord_head", 0.0) > 0,
        crosser_hidden_dim=config.get("crosser_hidden_dim", None) or None,
        disentangle=config.get("disentangle", True),
    )

    state_dict = _remap_legacy_vae_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: vae ({n_params:,} params)")
    val_top1 = checkpoint.get('val_top1')
    print(f"Best val Top-1 during training: {f'{val_top1:.2f}%' if val_top1 is not None else 'N/A'}")
    print()

    # Determine dataset and build manifest for case loading
    from config.defaults import DATASET_ROOTS
    dataset_name = args.dataset or config.get("dataset", "remind")

    # Resolve data root: CLI > config > default
    if args.dataset_root:
        data_root = Path(args.dataset_root)
    elif args.data_dir:
        data_root = Path(args.data_dir)
    else:
        data_root = Path(DATASET_ROOTS.get(dataset_name, config.get("data_dir", "")))

    # Build manifest (handles _full suffix, splits, file paths)
    manifest_kwargs: dict = {}
    if dataset_name in ("remind", "resect", "brats"):
        # Priority: --use_full CLI flag > stored config value > per-pair config
        # --use_full means "use full volumes" i.e. use_crop=False
        if args.use_full:
            use_crop = False
        elif "use_crop" in config:
            use_crop = bool(config["use_crop"])
        else:
            # Multi-dataset models store use_crop per pair; check if any pair
            # matching our modalities uses crop (typically US-involving pairs)
            use_crop = False
            for ds_entry in config.get("datasets", []):
                for pair in ds_entry.get("pairs", []):
                    pair_mods = {pair.get("source"), pair.get("target")}
                    if set(modalities) == pair_mods and pair.get("use_crop", False):
                        use_crop = True
                        break
        manifest_kwargs["use_crop"] = use_crop
    manifest = build_manifest(dataset_name, data_root, modalities, **manifest_kwargs)

    # Load test/val cases from manifest
    eval_cases = load_cases_from_manifest(manifest, args.split)
    if args.zscore:
        from src.io.case_adapter import _zscore_normalize
        print("Applying volume-level z-score normalization...")
        for case in eval_cases:
            for mod_name in list(case.volumes.keys()):
                case.volumes[mod_name] = _zscore_normalize(case.volumes[mod_name])
    if args.max_cases is not None and len(eval_cases) > args.max_cases:
        eval_cases = eval_cases[:args.max_cases]
    test_case_ids = [c.case_id for c in eval_cases]
    print(f"Dataset: {dataset_name}, {args.split} cases ({len(test_case_ids)}): "
          f"{[_display_id(cid) for cid in test_case_ids]}")
    print()

    # Create extractor
    if args.two_d_only:
        allowed_2d = {"sift"}
        if config.get("descriptor") not in allowed_2d:
            print(
                f"Error: --two_d_only requires a 2D extractor ({', '.join(sorted(allowed_2d))}); "
                f"got: {config.get('descriptor')}"
            )
            sys.exit(1)
    # SIFT-specific options
    extractor_kwargs = {}
    if config.get("descriptor") == "sift":
        if config.get("rootsift", False):
            extractor_kwargs["rootsift"] = True
        if config.get("no_compute_orientation", False):
            extractor_kwargs["compute_orientation"] = False
        preprocess = config.get("preprocess", "none")
        # For multi-dataset models, check per-dataset preprocessing
        # (e.g. gaussian_sobel only for whu_opt_sar)
        eval_ds = args.dataset or config.get("dataset", "remind")
        for ds_entry in config.get("datasets", []):
            if ds_entry.get("name") == eval_ds:
                ds_pp = ds_entry.get("preprocess")
                if ds_pp is not None:
                    preprocess = ds_pp
                break
        if preprocess != "none":
            extractor_kwargs["preprocess"] = preprocess
    extractor = SIFTDescriptor(**extractor_kwargs)
    if extractor_kwargs:
        print(f"Extractor options: {extractor_kwargs}")

    # Validate pair_sampling compatibility with descriptor
    if pair_sampling == "sift_union" and config.get("descriptor") != "sift":
        print(
            "Error: pair_sampling='sift_union' requires descriptor='sift', "
            f"got: {config.get('descriptor')}"
        )
        sys.exit(1)
    if pair_sampling in {"2d_sift_ref", "2d_sift_dual"}:
        if not (hasattr(extractor, "detect_keypoints_2d") and hasattr(extractor, "compute_2d")):
            print(
                f"Error: pair_sampling={pair_sampling!r} requires an extractor with "
                f"detect_keypoints_2d() and compute_2d(), but {type(extractor).__name__} "
                f"does not support them."
            )
            sys.exit(1)

    if args.max_kpts_per_slice is not None:
        max_kpts_per_slice = int(args.max_kpts_per_slice)
    else:
        max_kpts_per_slice = int(config.get("max_kpts_per_slice", 500))
    slice_budget_target: Optional[int] = None
    if pair_sampling in {"2d_sift_ref", "2d_sift_dual"}:
        slice_budget_target = int((int(args.n_samples) + max_kpts_per_slice - 1) // max_kpts_per_slice)
        print(
            "2D slice budget: "
            f"ceil(n_samples / max_kpts_per_slice) = {slice_budget_target} "
            f"(n_samples={int(args.n_samples)}, max_kpts_per_slice={max_kpts_per_slice})"
        )
    print()

    # Create test-time adapter (if specified)
    adapter = None
    if args.adapter and args.adapter != "none":
        if args.adapter != "inlier_ridge":
            print(
                f"Error: adapter={args.adapter!r} is unsupported in CrossFeat. "
                "Only adapter='inlier_ridge' is supported."
            )
            sys.exit(1)
        adapter = InlierRidgeAdapter(
            n_iterations=args.adapter_n_iterations,
            ridge_alpha=args.adapter_ridge_alpha,
            ransac_threshold=args.inlier_threshold,
        )
        print(f"Test-time adapter: {adapter}")
        print()

    # Create baseline runner (once) if requested.
    baseline_runner, baseline_error = _create_nocross_baseline(args)
    if baseline_error:
        print(f"ERROR: Failed to initialize baseline {args.baseline}: {baseline_error}")
        if args.baseline == "lightglue":
            print("  Tip: install with `pip install git+https://github.com/cvg/LightGlue.git`")
        print()
    elif baseline_runner is not None and hasattr(baseline_runner, "warmup"):
        # Eagerly load baseline model to avoid loading messages during evaluation table
        baseline_runner.warmup()

    # Evaluate each case
    results: List[EvaluationResult] = []
    baseline_results: List[EvaluationResult] = []

    # Visualization caches (populated when --viz is enabled)
    viz_enabled = args.viz or args.viz_compare_cross or args.viz_compare_tta
    viz_cache: Dict[str, VisualizationData] = {}
    viz_cache_no_cross: Dict[str, VisualizationData] = {}
    viz_cache_no_tta: Dict[str, VisualizationData] = {}
    viz_baseline_enabled = bool(args.viz_baseline) and args.baseline != "none"
    viz_cache_baseline: Dict[str, VisualizationData] = {}
    viz_cases_processed = 0
    viz_baseline_cases_processed = 0
    vol_depths: list[int] = []
    crossfeat_slices_used: list[int] = []
    cases_not_found: list[str] = []
    cases_failed_to_load: list[str] = []

    requested_eval_workers = max(1, int(args.eval_workers))
    parallel_case_eval = (
        requested_eval_workers > 1
        and not viz_enabled
        and not viz_baseline_enabled  # block on ANY viz path
        and adapter is None           # adapters hold per-case mutable state
    )
    effective_eval_workers = 1
    if parallel_case_eval:
        effective_eval_workers = min(requested_eval_workers, len(eval_cases)) if eval_cases else 1
        clone_fn = getattr(extractor, "clone", None)
        if not callable(clone_fn):
            print(
                f"Parallel eval requested (--eval_workers={requested_eval_workers}) but "
                f"extractor {type(extractor).__name__} has no clone(); falling back to sequential."
            )
            parallel_case_eval = False
        else:
            # GPU extractors (SuperPoint, DeDoDe, etc.) don't benefit from
            # threading: Python's GIL serialises CPU work and duplicating
            # models on the same GPU adds contention rather than parallelism.
            ext_device = str(getattr(extractor, "device", "cpu")).lower()
            if "cuda" in ext_device:
                print(
                    f"Parallel eval requested (--eval_workers={requested_eval_workers}) but "
                    f"extractor {type(extractor).__name__} runs on GPU ({ext_device}); "
                    "multi-threaded GPU inference is slower — falling back to sequential."
                )
                parallel_case_eval = False
            else:
                try:
                    probe_clone = clone_fn()
                    if probe_clone is extractor:
                        print(
                            f"Parallel eval requested (--eval_workers={requested_eval_workers}) but "
                            "extractor clone() returned self; falling back to sequential."
                        )
                        parallel_case_eval = False
                except Exception as e:
                    print(
                        f"Parallel eval requested (--eval_workers={requested_eval_workers}) but "
                        f"extractor clone() failed ({type(e).__name__}: {e}); falling back to sequential."
                        )
                    parallel_case_eval = False

    if requested_eval_workers > 1 and not parallel_case_eval:
        print(
            "--eval_workers is only used when running without visualization "
            "(--viz/--viz_baseline) and without adapters. Baseline comparison "
            "is supported in parallel mode (concurrency set via --baseline_workers)."
        )

    print()
    print(f"{'Case':<30} {'Top-1':>8} {'#Match':>7} {'#Inl':>6} {'Prec':>7} {'Recall':>7} {'Cov':>6} {'TRE':>7} {'Time':>7}")
    print("-" * 120)

    # Baseline always evaluates on the same slices CrossFeat used (from result.slices_used).
    has_baseline = baseline_runner is not None and args.baseline != "none"

    if parallel_case_eval:
        tls = threading.local()

        # Parallelism model: CrossFeat descriptor extraction runs in parallel across
        # workers (each with a cloned extractor), while baseline GPU inference is
        # gated by a Semaphore allowing limited concurrency (--baseline_workers).
        # Net win: overlaps CF extraction of case N+1 with baseline eval of case N,
        # and allows multiple baseline evaluations to overlap on GPU.
        baseline_sem: threading.Semaphore | None = None
        if has_baseline:
            bw = int(args.baseline_workers) if args.baseline_workers > 0 else max(1, effective_eval_workers // 4)
            baseline_sem = threading.Semaphore(bw)
            print(f"Parallel eval: {effective_eval_workers} workers, {bw} baseline slots")

        def _eval_worker(case_idx: int, case_obj: CaseData):
            case_id_local = case_obj.case_id
            if not all(m in case_obj.volumes for m in modalities):
                return case_idx, case_id_local, None, "missing_modalities", None, None, None

            if not hasattr(tls, "extractor"):
                tls.extractor = extractor.clone()
            ext = tls.extractor

            z_slices_preselected_local: Optional[List[int]] = None
            _need_slice_preselect = (
                args.two_d_only
                and (args.slice_sampling == "uniform_random" or args.num_slices is not None)
            )
            if _need_slice_preselect:
                vol_a_local = case_obj.volumes[modalities[0]]
                vol_b_local = case_obj.volumes[modalities[1]]
                base_margin_local = 15
                erode_margin_local = 5

                abs_a_local = np.abs(vol_a_local)
                abs_b_local = np.abs(vol_b_local)
                max_a_local = float(np.max(abs_a_local))
                max_b_local = float(np.max(abs_b_local))
                if max_a_local > 0.0 and max_b_local > 0.0:
                    mask_a_local = _infer_signal_mask(vol_a_local, abs_volume=abs_a_local, max_abs=max_a_local)
                    mask_b_local = _infer_signal_mask(vol_b_local, abs_volume=abs_b_local, max_abs=max_b_local)
                    mask_a_local &= abs_a_local > 0.01 * max_a_local
                    mask_b_local &= abs_b_local > 0.01 * max_b_local
                    joint_mask_local = mask_a_local & mask_b_local
                    joint_mask_local = safe_preprocess_roi(
                        joint_mask_local,
                        erode_iterations=erode_margin_local,
                        base_margin=base_margin_local,
                    )

                    shape_local = np.array(vol_a_local.shape)
                    crop_min_local, crop_max_local = compute_safe_crop_bounds(
                        tuple(int(s) for s in shape_local),
                        base_margin_local,
                    )

                    num_slices_local = args.num_slices if args.num_slices is not None else 999999
                    if args.slice_sampling == "uniform_random":
                        case_seed_local = int(args.seed) + int(hashlib.md5(case_id_local.encode()).hexdigest(), 16) % 10000
                        rng_local = np.random.RandomState(case_seed_local)
                        z_slices_preselected_local = select_slices_uniform_random(
                            roi_mask=joint_mask_local,
                            crop_min=crop_min_local,
                            crop_max=crop_max_local,
                            n_slices=num_slices_local,
                            rng=rng_local,
                            min_roi_area=100,
                        )
                    else:
                        z_slices_preselected_local = select_slices_by_roi_area(
                            roi_mask=joint_mask_local,
                            crop_min=crop_min_local,
                            crop_max=crop_max_local,
                            n_slices=num_slices_local,
                            min_roi_area=100,
                        )

            try:
                result_local = evaluate_case(
                    case=case_obj,
                    modalities=modalities,
                    extractor=ext,
                    pca=pca,
                    model=model,
                    device=args.device,
                    rotate_deg=args.rotate_deg,
                    translate_mm=args.translate_mm,
                    affine_scale=args.affine_scale,
                    affine_shear_deg=args.affine_shear_deg,
                    deterministic_transform=args.deterministic_transform,
                    n_samples=args.n_samples,
                    inlier_threshold=args.inlier_threshold,
                    seed=args.seed,
                    adapter=None,
                    two_d_only=args.two_d_only,
                    pair_sampling=pair_sampling,
                    max_kpts_per_slice=max_kpts_per_slice,
                    z_slices=z_slices_preselected_local,
                    mod_a_idx=mod_a_idx,
                    mod_b_idx=mod_b_idx,
                    ratio_thresh=args.ratio_thresh,
                    match_per_slice=args.match_per_slice,
                )
            except Exception as e:
                result_local = _make_error_result(case_id_local, f"{type(e).__name__}: {e}", 0.0)

            vol_depth_local = int(case_obj.volumes[modalities[0]].shape[0])

            # --- Baseline evaluation (concurrency limited via baseline_sem) ---
            baseline_result_local = None
            slices_used_local = None
            if baseline_runner is not None and result_local is not None and not result_local.error:
                # Always sync slices: baseline evaluates on the same slices CrossFeat used.
                bl_z_slices = result_local.slices_used if result_local.slices_used else None
                if bl_z_slices:
                    slices_used_local = len(bl_z_slices)

                with baseline_sem:
                    baseline_out = baseline_runner.evaluate_case(
                        case=case_obj,
                        modalities=modalities,
                        extractor=ext,
                        pca=pca,
                        device=args.device,
                        rotate_deg=args.rotate_deg,
                        translate_mm=args.translate_mm,
                        affine_scale=args.affine_scale,
                        affine_shear_deg=args.affine_shear_deg,
                        deterministic_transform=args.deterministic_transform,
                        n_samples_for_transform=args.n_samples,
                        inlier_threshold=args.inlier_threshold,
                        seed=args.seed,
                        adapter=None,
                        return_viz_data=False,
                        two_d_only=args.two_d_only,
                        pair_sampling=pair_sampling,
                        max_kpts_per_slice=max_kpts_per_slice,
                        z_slices=bl_z_slices,
                        ratio_thresh=args.ratio_thresh,
                    )
                if isinstance(baseline_out, tuple):
                    baseline_result_local = baseline_out[0]
                else:
                    baseline_result_local = baseline_out
            elif baseline_runner is None and args.baseline != "none":
                baseline_result_local = _make_error_result(
                    case_id_local,
                    baseline_error or f"Baseline {args.baseline} unavailable",
                    0.0,
                )

            return case_idx, case_id_local, result_local, None, vol_depth_local, baseline_result_local, slices_used_local

        ordered_results: list[Optional[tuple]] = [None] * len(eval_cases)
        with ThreadPoolExecutor(max_workers=effective_eval_workers) as pool:
            futures = {
                pool.submit(_eval_worker, idx, case): idx
                for idx, case in enumerate(eval_cases)
            }
            for future in as_completed(futures):
                fut_idx = futures[future]
                try:
                    worker_result = future.result()
                    case_idx = worker_result[0]
                    ordered_results[case_idx] = worker_result[1:]  # drop case_idx
                except Exception as exc:
                    print(f"Case {fut_idx} evaluation failed: {type(exc).__name__}: {exc}")
                    ordered_results[fut_idx] = (f"Case{fut_idx}", None, str(exc), None, None, None)

        for item in ordered_results:
            if item is None:
                continue
            case_id, result, err, depth, bl_result, sl_used = item
            if err is not None or result is None:
                cases_failed_to_load.append(case_id)
                continue
            if depth is not None:
                vol_depths.append(depth)

            results.append(result)
            short_id = _display_id(case_id)
            if result.error:
                print(f"{short_id:<30} ERROR: {result.error}")
            else:
                slices_str = f"  slices={result.slices_used}" if result.slices_used else ""
                print(
                    f"{short_id:<30} {result.top1_acc:>8.3f} {result.num_matches:>7d} "
                    f"{result.num_inliers:>6d} {result.inlier_ratio:>7.3f} {result.recall:>7.3f} "
                    f"{result.spatial_coverage:>6.3f} {result.tre_mean:>7.2f} {result.time_s:>6.1f}s{slices_str}"
                )

            # Baseline result from parallel worker
            if bl_result is not None:
                baseline_results.append(bl_result)
                if sl_used is not None:
                    crossfeat_slices_used.append(sl_used)
                baseline_short = {
                    "nocross": "NC", "lightglue": "LG",
                    "disk_lg": "DLG", "aliked_lg": "ALG", "sift_lg": "SLG",
                    "matchanything": "MA",
                    "minima_lg": "MLG", "minima_loftr": "ML", "minima_roma": "MR",
                }.get(str(args.baseline).lower(), str(args.baseline))
                case_label = f"{short_id}({baseline_short})"
                if bl_result.error:
                    print(f"{case_label:<30} ERROR: {bl_result.error}")
                else:
                    slices_str = f"  slices={bl_result.slices_used}" if bl_result.slices_used else ""
                    print(
                        f"{case_label:<30} {bl_result.top1_acc:>8.3f} {bl_result.num_matches:>7d} "
                        f"{bl_result.num_inliers:>6d} {bl_result.inlier_ratio:>7.3f} {bl_result.recall:>7.3f} "
                        f"{bl_result.spatial_coverage:>6.3f} {bl_result.tre_mean:>7.2f} {bl_result.time_s:>6.1f}s{slices_str}"
                    )

        # Skip the sequential loop below (already processed in parallel path).
        eval_cases = []

    for case in eval_cases:
        case_id = case.case_id
        if not all(m in case.volumes for m in modalities):
            cases_failed_to_load.append(case_id)
            continue

        vol_depths.append(int(case.volumes[modalities[0]].shape[0]))

        # Pre-select slices (uniform_random always, sift when --num_slices given)
        z_slices_preselected: Optional[List[int]] = None
        _need_slice_preselect = (
            args.two_d_only
            and (args.slice_sampling == "uniform_random" or args.num_slices is not None)
        )
        if _need_slice_preselect:
            vol_a = case.volumes[modalities[0]]
            vol_b = case.volumes[modalities[1]]
            base_margin = 15
            erode_margin = 5

            # Compute joint ROI mask
            abs_a = np.abs(vol_a)
            abs_b = np.abs(vol_b)
            max_a = float(np.max(abs_a))
            max_b = float(np.max(abs_b))
            if max_a > 0.0 and max_b > 0.0:
                mask_a = _infer_signal_mask(vol_a, abs_volume=abs_a, max_abs=max_a)
                mask_b = _infer_signal_mask(vol_b, abs_volume=abs_b, max_abs=max_b)
                mask_a &= abs_a > 0.01 * max_a
                mask_b &= abs_b > 0.01 * max_b
                joint_mask = mask_a & mask_b
                joint_mask = safe_preprocess_roi(joint_mask, erode_iterations=erode_margin, base_margin=base_margin)

                shape = np.array(vol_a.shape)
                crop_min, crop_max = compute_safe_crop_bounds(tuple(int(s) for s in shape), base_margin)

                num_slices = args.num_slices if args.num_slices is not None else 999999
                if args.slice_sampling == "uniform_random":
                    case_seed = int(args.seed) + int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 10000
                    rng = np.random.RandomState(case_seed)
                    z_slices_preselected = select_slices_uniform_random(
                        roi_mask=joint_mask,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        n_slices=num_slices,
                        rng=rng,
                        min_roi_area=100,
                    )
                else:
                    z_slices_preselected = select_slices_by_roi_area(
                        roi_mask=joint_mask,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        n_slices=num_slices,
                        min_roi_area=100,
                    )

        # Determine if we need viz data for this case
        need_viz = viz_enabled and viz_cases_processed < args.viz_max_cases
        need_viz_baseline = viz_baseline_enabled and viz_baseline_cases_processed < args.viz_max_cases
        need_case_viz_data = bool(need_viz)

        viz_data = None
        if need_case_viz_data:
            result, viz_data = evaluate_case(
                case=case,
                modalities=modalities,
                extractor=extractor,
                pca=pca,
                model=model,
                device=args.device,
                rotate_deg=args.rotate_deg,
                translate_mm=args.translate_mm,
                affine_scale=args.affine_scale,
                affine_shear_deg=args.affine_shear_deg,
                deterministic_transform=args.deterministic_transform,
                n_samples=args.n_samples,
                inlier_threshold=args.inlier_threshold,
                seed=args.seed,
                adapter=adapter,
                return_viz_data=True,
                two_d_only=args.two_d_only,
                pair_sampling=pair_sampling,
                max_kpts_per_slice=max_kpts_per_slice,
                z_slices=z_slices_preselected,
                mod_a_idx=mod_a_idx,
                mod_b_idx=mod_b_idx,
                ratio_thresh=args.ratio_thresh,
                match_per_slice=args.match_per_slice,
            )
            if need_viz and viz_data is not None:
                viz_cache[case_id] = viz_data
                viz_cases_processed += 1

            # Run without crossing for comparison (if requested)
            if need_viz and args.viz_compare_cross and viz_data is not None:
                _, viz_data_no_cross = evaluate_case_no_crossing(
                    case=case,
                    modalities=modalities,
                    extractor=extractor,
                    pca=pca,
                    device=args.device,
                    rotate_deg=args.rotate_deg,
                    translate_mm=args.translate_mm,
                    affine_scale=args.affine_scale,
                    affine_shear_deg=args.affine_shear_deg,
                    deterministic_transform=args.deterministic_transform,
                    n_samples=args.n_samples,
                    inlier_threshold=args.inlier_threshold,
                    seed=args.seed,
                    two_d_only=args.two_d_only,
                    pair_sampling=pair_sampling,
                    max_kpts_per_slice=max_kpts_per_slice,
                    z_slices=z_slices_preselected,
                    ratio_thresh=args.ratio_thresh,
                    match_per_slice=args.match_per_slice,
                )
                if viz_data_no_cross is not None:
                    viz_cache_no_cross[case_id] = viz_data_no_cross

            # Run without TTA for comparison (if requested and TTA is enabled)
            if need_viz and args.viz_compare_tta and adapter is not None and viz_data is not None:
                _, viz_data_no_tta = evaluate_case(
                    case=case,
                    modalities=modalities,
                    extractor=extractor,
                    pca=pca,
                    model=model,
                    device=args.device,
                    rotate_deg=args.rotate_deg,
                    translate_mm=args.translate_mm,
                    affine_scale=args.affine_scale,
                    affine_shear_deg=args.affine_shear_deg,
                    deterministic_transform=args.deterministic_transform,
                    n_samples=args.n_samples,
                    inlier_threshold=args.inlier_threshold,
                    seed=args.seed,
                    adapter=None,
                    return_viz_data=True,
                    two_d_only=args.two_d_only,
                    pair_sampling=pair_sampling,
                    max_kpts_per_slice=max_kpts_per_slice,
                    z_slices=z_slices_preselected,
                    mod_a_idx=mod_a_idx,
                    mod_b_idx=mod_b_idx,
                    ratio_thresh=args.ratio_thresh,
                    match_per_slice=args.match_per_slice,
                )
                if viz_data_no_tta is not None:
                    viz_cache_no_tta[case_id] = viz_data_no_tta
        else:
            result = evaluate_case(
                case=case,
                modalities=modalities,
                extractor=extractor,
                pca=pca,
                model=model,
                device=args.device,
                rotate_deg=args.rotate_deg,
                translate_mm=args.translate_mm,
                affine_scale=args.affine_scale,
                affine_shear_deg=args.affine_shear_deg,
                deterministic_transform=args.deterministic_transform,
                n_samples=args.n_samples,
                inlier_threshold=args.inlier_threshold,
                seed=args.seed,
                adapter=adapter,
                two_d_only=args.two_d_only,
                pair_sampling=pair_sampling,
                max_kpts_per_slice=max_kpts_per_slice,
                z_slices=z_slices_preselected,
                mod_a_idx=mod_a_idx,
                mod_b_idx=mod_b_idx,
                ratio_thresh=args.ratio_thresh,
                match_per_slice=args.match_per_slice,
            )

        results.append(result)

        short_id = _display_id(case_id)
        if result.error:
            print(f"{short_id:<30} ERROR: {result.error}")
        else:
            slices_str = f"  slices={result.slices_used}" if result.slices_used else ""
            print(f"{short_id:<30} {result.top1_acc:>8.3f} {result.num_matches:>7d} "
                  f"{result.num_inliers:>6d} {result.inlier_ratio:>7.3f} {result.recall:>7.3f} "
                  f"{result.spatial_coverage:>6.3f} {result.tre_mean:>7.2f} {result.time_s:>6.1f}s{slices_str}")

        if args.baseline != "none":
            # Always sync: baseline evaluates on the same slices CrossFeat used.
            z_slices = result.slices_used if result.slices_used else None
            if z_slices:
                crossfeat_slices_used.append(len(z_slices))

            baseline_viz = None
            if baseline_runner is None:
                baseline = _make_error_result(case_id, baseline_error or f"Baseline {args.baseline} unavailable", 0.0)
            else:
                baseline_out = baseline_runner.evaluate_case(
                    case=case,
                    modalities=modalities,
                    extractor=extractor,
                    pca=pca,
                    device=args.device,
                    rotate_deg=args.rotate_deg,
                    translate_mm=args.translate_mm,
                    affine_scale=args.affine_scale,
                    affine_shear_deg=args.affine_shear_deg,
                    deterministic_transform=args.deterministic_transform,
                    n_samples_for_transform=args.n_samples,
                    inlier_threshold=args.inlier_threshold,
                    seed=args.seed,
                    adapter=None,  # Adapter refines crossing model; baselines do their own matching
                    return_viz_data=need_viz_baseline,
                    two_d_only=args.two_d_only,
                    pair_sampling=pair_sampling,
                    max_kpts_per_slice=max_kpts_per_slice,
                    z_slices=z_slices,
                    ratio_thresh=args.ratio_thresh,
                )
                if isinstance(baseline_out, tuple):
                    baseline, baseline_viz = baseline_out
                else:
                    baseline, baseline_viz = baseline_out, None

            if baseline_viz is not None:
                viz_cache_baseline[case_id] = baseline_viz
                viz_baseline_cases_processed += 1

            baseline_results.append(baseline)
            baseline_short = {"nocross": "NC", "lightglue": "LG", "disk_lg": "DLG", "aliked_lg": "ALG", "sift_lg": "SLG", "matchanything": "MA", "minima_lg": "MLG", "minima_loftr": "ML", "minima_roma": "MR"}.get(str(args.baseline).lower(), str(args.baseline))
            case_label = f"{short_id}({baseline_short})"

            if baseline.error:
                print(f"{case_label:<30} ERROR: {baseline.error}")
            else:
                slices_str = f"  slices={baseline.slices_used}" if baseline.slices_used else ""
                print(f"{case_label:<30} {baseline.top1_acc:>8.3f} {baseline.num_matches:>7d} "
                      f"{baseline.num_inliers:>6d} {baseline.inlier_ratio:>7.3f} {baseline.recall:>7.3f} "
                      f"{baseline.spatial_coverage:>6.3f} {baseline.tre_mean:>7.2f} {baseline.time_s:>6.1f}s{slices_str}")

    # Summary
    valid_results = [r for r in results if not r.error]
    valid_baseline_results = [r for r in baseline_results if not r.error]

    # Print skipped cases summary
    if cases_not_found or cases_failed_to_load:
        print()
        if cases_not_found:
            print(f"Not found: {[_display_id(c) for c in cases_not_found]}")
        if cases_failed_to_load:
            print(f"Failed to load: {[_display_id(c) for c in cases_failed_to_load]}")

    def _nanmean_or_none(vals: list[float]) -> Optional[float]:
        """Return nanmean of *vals*, or None if all values are NaN."""
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return None
        return float(np.nanmean(arr))

    print()
    print("=" * 120)
    print("SUMMARY (CrossFeat)")
    print("=" * 120)

    if valid_results:
        top1_vals = [r.top1_acc for r in valid_results]
        top5_vals = [r.top5_acc for r in valid_results]
        top10_vals = [r.top10_acc for r in valid_results]
        # Exclude 0-match cases from precision: 0 matches means precision
        # is undefined (not 0), so including them drags down the mean.
        ir_vals = [r.inlier_ratio for r in valid_results if r.num_matches > 0]
        if not ir_vals:
            ir_vals = [0.0]
        recall_vals = [r.recall for r in valid_results]
        cov_vals = [r.spatial_coverage for r in valid_results]
        tre_vals = [r.tre_mean for r in valid_results if np.isfinite(r.tre_mean)]
        cos_vals = [r.cosine_sim for r in valid_results]
        total_matches = sum(r.num_matches for r in valid_results)
        total_inliers = sum(r.num_inliers for r in valid_results)
        total_slices = sum(len(r.slices_used) for r in valid_results if r.slices_used)

        # Collect SR and AUC values for summary.
        # NaN values (from cases with no verifiable depth projection) are
        # excluded via np.nanmean so they don't drag down the average.
        sr_1px_vals = [r.sr_1px for r in valid_results]
        sr_3px_vals = [r.sr_3px for r in valid_results]
        sr_5px_vals = [r.sr_5px for r in valid_results]
        sr_10px_vals = [r.sr_10px for r in valid_results]
        auc_1px_vals = [r.auc_1px for r in valid_results]
        auc_3px_vals = [r.auc_3px for r in valid_results]
        auc_5px_vals = [r.auc_5px for r in valid_results]
        auc_10px_vals = [r.auc_10px for r in valid_results]

        # TRE unit: "px" for 2D-only mode (axial slice matching), "vox" otherwise
        tre_unit = "px" if args.two_d_only else "vox"

        print(f"  Cases evaluated: {len(valid_results)}")
        print(f"  Top-1 Accuracy:  {np.mean(top1_vals)*100:.2f}% ± {np.std(top1_vals)*100:.2f}%")
        print(f"  Top-5 Accuracy:  {np.mean(top5_vals)*100:.2f}% ± {np.std(top5_vals)*100:.2f}%")
        print(f"  Precision (IR):  {np.mean(ir_vals)*100:.2f}% ± {np.std(ir_vals)*100:.2f}%")
        print(f"  Recall:          {np.mean(recall_vals)*100:.2f}% ± {np.std(recall_vals)*100:.2f}%")
        print(f"  Spatial Cov:     {np.mean(cov_vals):.3f} ± {np.std(cov_vals):.3f}")
        print(f"  Total Matches:   {total_matches} ({total_inliers} inliers)")
        if tre_vals:
            print(f"  Mean TRE:        {np.mean(tre_vals):.2f} ± {np.std(tre_vals):.2f} {tre_unit}")
        print(f"  Cosine Sim:      {np.mean(cos_vals):.3f} ± {np.std(cos_vals):.3f}")
        print()
        # SR @ thresholds (aggregated as percentage of successful cases)
        # Use nanmean to skip cases with unverifiable depth projection (NaN SR).
        print(f"  Success Rate:    SR@1px: {np.nanmean(sr_1px_vals)*100:.1f}% | "
              f"SR@3px: {np.nanmean(sr_3px_vals)*100:.1f}% | "
              f"SR@5px: {np.nanmean(sr_5px_vals)*100:.1f}% | "
              f"SR@10px: {np.nanmean(sr_10px_vals)*100:.1f}%")
        # AUC @ thresholds (aggregated as mean AUC)
        print(f"  AUC:             AUC@1px: {np.nanmean(auc_1px_vals):.3f} | "
              f"AUC@3px: {np.nanmean(auc_3px_vals):.3f} | "
              f"AUC@5px: {np.nanmean(auc_5px_vals):.3f} | "
              f"AUC@10px: {np.nanmean(auc_10px_vals):.3f}")
        print()

        if args.rotate_deg == 0 and args.translate_mm == 0 and args.affine_scale == 0 and args.affine_shear_deg == 0:
            print("  (Evaluated on ALIGNED data)")
        else:
            parts = [
                f"±{args.rotate_deg}° rotation",
                f"±{args.translate_mm}mm translation",
            ]
            if args.affine_scale > 0:
                parts.append(f"scale±{args.affine_scale:.2f}")
            if args.affine_shear_deg > 0:
                parts.append(f"shear±{args.affine_shear_deg:.1f}°")
            print(f"  (Evaluated on MISALIGNED data: {', '.join(parts)})")

        if vol_depths:
            print(
                "  Axial slices (Z): "
                f"min={int(np.min(vol_depths))}, mean={float(np.mean(vol_depths)):.1f}, max={int(np.max(vol_depths))}"
            )
        if crossfeat_slices_used and slice_budget_target is not None:
            print(
                "  CrossFeat slices used (2d_sift_*): "
                f"mean={float(np.mean(crossfeat_slices_used)):.2f} ± {float(np.std(crossfeat_slices_used)):.2f} "
                f"(target≈{slice_budget_target})"
            )

        # Verdict
        avg_top1 = np.mean(top1_vals)
        print()
        if avg_top1 >= 0.7:
            print(f"  ✅ EXCELLENT: Top-1 = {avg_top1*100:.1f}%")
        elif avg_top1 >= 0.5:
            print(f"  ✓ GOOD: Top-1 = {avg_top1*100:.1f}%")
        elif avg_top1 >= 0.2:
            print(f"  ⚠️  FAIR: Top-1 = {avg_top1*100:.1f}%")
        else:
            print(f"  ❌ POOR: Top-1 = {avg_top1*100:.1f}%")

        # Prepare summary dict
        summary = {
            "num_cases": len(valid_results),
            "top1_acc_mean": float(np.mean(top1_vals)),
            "top1_acc_std": float(np.std(top1_vals)),
            "top5_acc_mean": float(np.mean(top5_vals)),
            "top5_acc_std": float(np.std(top5_vals)),
            "top10_acc_mean": float(np.mean(top10_vals)),
            "top10_acc_std": float(np.std(top10_vals)),
            "precision_mean": float(np.mean(ir_vals)),
            "precision_std": float(np.std(ir_vals)),
            "recall_mean": float(np.mean(recall_vals)),
            "recall_std": float(np.std(recall_vals)),
            "spatial_coverage_mean": float(np.mean(cov_vals)),
            "spatial_coverage_std": float(np.std(cov_vals)),
            "total_matches": total_matches,
            "total_inliers": total_inliers,
            "total_slices": total_slices,
            "tre_mean": float(np.mean(tre_vals)) if tre_vals else None,
            "tre_std": float(np.std(tre_vals)) if tre_vals else None,
            "cosine_sim_mean": float(np.mean(cos_vals)),
            "cosine_sim_std": float(np.std(cos_vals)),
            # SR @ thresholds (Success Rate: fraction of successful cases)
            # None when all cases are unverifiable (NaN SR).
            "sr_1px": _nanmean_or_none(sr_1px_vals),
            "sr_3px": _nanmean_or_none(sr_3px_vals),
            "sr_5px": _nanmean_or_none(sr_5px_vals),
            "sr_10px": _nanmean_or_none(sr_10px_vals),
            # AUC @ thresholds (mean AUC across cases)
            "auc_1px": _nanmean_or_none(auc_1px_vals),
            "auc_3px": _nanmean_or_none(auc_3px_vals),
            "auc_5px": _nanmean_or_none(auc_5px_vals),
            "auc_10px": _nanmean_or_none(auc_10px_vals),
        }
    else:
        print("  No valid results")
        summary = {"error": "No valid results"}

    baseline_summary: Optional[Dict[str, Any]] = None
    if args.baseline != "none":
        print()
        print("=" * 120)
        print(f"SUMMARY (Baseline: {args.baseline})")
        print("=" * 120)

        if valid_baseline_results:
            top1_vals = [r.top1_acc for r in valid_baseline_results]
            top5_vals = [r.top5_acc for r in valid_baseline_results]
            top10_vals = [r.top10_acc for r in valid_baseline_results]
            # Exclude 0-match cases from precision: 0 matches means precision
            # is undefined (not 0), so including them drags down the mean.
            ir_vals = [r.inlier_ratio for r in valid_baseline_results if r.num_matches > 0]
            if not ir_vals:
                ir_vals = [0.0]
            recall_vals = [r.recall for r in valid_baseline_results]
            cov_vals = [r.spatial_coverage for r in valid_baseline_results]
            tre_vals = [r.tre_mean for r in valid_baseline_results if np.isfinite(r.tre_mean)]
            cos_vals = [r.cosine_sim for r in valid_baseline_results]
            total_matches = sum(r.num_matches for r in valid_baseline_results)
            total_inliers = sum(r.num_inliers for r in valid_baseline_results)
            total_slices = sum(len(r.slices_used) for r in valid_baseline_results if r.slices_used)

            # Collect SR and AUC values for baseline summary
            bl_sr_1px_vals = [r.sr_1px for r in valid_baseline_results]
            bl_sr_3px_vals = [r.sr_3px for r in valid_baseline_results]
            bl_sr_5px_vals = [r.sr_5px for r in valid_baseline_results]
            bl_sr_10px_vals = [r.sr_10px for r in valid_baseline_results]
            bl_auc_1px_vals = [r.auc_1px for r in valid_baseline_results]
            bl_auc_3px_vals = [r.auc_3px for r in valid_baseline_results]
            bl_auc_5px_vals = [r.auc_5px for r in valid_baseline_results]
            bl_auc_10px_vals = [r.auc_10px for r in valid_baseline_results]

            # TRE unit: "px" for 2D-only mode (axial slice matching), "vox" otherwise
            tre_unit = "px" if args.two_d_only else "vox"

            print(f"  Cases evaluated: {len(valid_baseline_results)}")
            print(f"  Top-1 Accuracy:  {np.mean(top1_vals)*100:.2f}% ± {np.std(top1_vals)*100:.2f}%")
            print(f"  Top-5 Accuracy:  {np.mean(top5_vals)*100:.2f}% ± {np.std(top5_vals)*100:.2f}%")
            print(f"  Precision (IR):  {np.mean(ir_vals)*100:.2f}% ± {np.std(ir_vals)*100:.2f}%")
            print(f"  Recall:          {np.mean(recall_vals)*100:.2f}% ± {np.std(recall_vals)*100:.2f}%")
            print(f"  Spatial Cov:     {np.mean(cov_vals):.3f} ± {np.std(cov_vals):.3f}")
            print(f"  Total Matches:   {total_matches} ({total_inliers} inliers)")
            if tre_vals:
                print(f"  Mean TRE:        {np.mean(tre_vals):.2f} ± {np.std(tre_vals):.2f} {tre_unit}")
            print(f"  Cosine Sim:      {np.mean(cos_vals):.3f} ± {np.std(cos_vals):.3f}")
            print()
            # SR @ thresholds (aggregated as percentage of successful cases)
            print(f"  Success Rate:    SR@1px: {np.nanmean(bl_sr_1px_vals)*100:.1f}% | "
                  f"SR@3px: {np.nanmean(bl_sr_3px_vals)*100:.1f}% | "
                  f"SR@5px: {np.nanmean(bl_sr_5px_vals)*100:.1f}% | "
                  f"SR@10px: {np.nanmean(bl_sr_10px_vals)*100:.1f}%")
            # AUC @ thresholds (aggregated as mean AUC)
            print(f"  AUC:             AUC@1px: {np.nanmean(bl_auc_1px_vals):.3f} | "
                  f"AUC@3px: {np.nanmean(bl_auc_3px_vals):.3f} | "
                  f"AUC@5px: {np.nanmean(bl_auc_5px_vals):.3f} | "
                  f"AUC@10px: {np.nanmean(bl_auc_10px_vals):.3f}")
            print()

            baseline_summary = {
                "name": args.baseline,
                "num_cases": len(valid_baseline_results),
                "top1_acc_mean": float(np.mean(top1_vals)),
                "top1_acc_std": float(np.std(top1_vals)),
                "top5_acc_mean": float(np.mean(top5_vals)),
                "top5_acc_std": float(np.std(top5_vals)),
                "top10_acc_mean": float(np.mean(top10_vals)),
                "top10_acc_std": float(np.std(top10_vals)),
                "precision_mean": float(np.mean(ir_vals)),
                "precision_std": float(np.std(ir_vals)),
                "recall_mean": float(np.mean(recall_vals)),
                "recall_std": float(np.std(recall_vals)),
                "spatial_coverage_mean": float(np.mean(cov_vals)),
                "spatial_coverage_std": float(np.std(cov_vals)),
                "total_matches": total_matches,
                "total_inliers": total_inliers,
                "total_slices": total_slices,
                "tre_mean": float(np.mean(tre_vals)) if tre_vals else None,
                "tre_std": float(np.std(tre_vals)) if tre_vals else None,
                "cosine_sim_mean": float(np.mean(cos_vals)),
                "cosine_sim_std": float(np.std(cos_vals)),
                # SR @ thresholds (Success Rate: fraction of successful cases)
                # None when all cases are unverifiable (NaN SR).
                "sr_1px": _nanmean_or_none(bl_sr_1px_vals),
                "sr_3px": _nanmean_or_none(bl_sr_3px_vals),
                "sr_5px": _nanmean_or_none(bl_sr_5px_vals),
                "sr_10px": _nanmean_or_none(bl_sr_10px_vals),
                # AUC @ thresholds (mean AUC across cases)
                "auc_1px": _nanmean_or_none(bl_auc_1px_vals),
                "auc_3px": _nanmean_or_none(bl_auc_3px_vals),
                "auc_5px": _nanmean_or_none(bl_auc_5px_vals),
                "auc_10px": _nanmean_or_none(bl_auc_10px_vals),
            }
        else:
            print("  No valid baseline results")
            baseline_summary = {"name": args.baseline, "error": "No valid results"}

    print("=" * 70)

    # Generate visualizations if enabled
    if (viz_enabled and viz_cache) or (viz_baseline_enabled and viz_cache_baseline):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        aug_parts: list[str] = []
        if args.rotate_deg > 0:
            aug_parts.append(f"rot{args.rotate_deg}")
        if args.translate_mm > 0:
            aug_parts.append(f"trans{args.translate_mm}")
        if args.affine_scale > 0:
            aug_parts.append(f"scale{args.affine_scale}")
        if args.affine_shear_deg > 0:
            aug_parts.append(f"shear{args.affine_shear_deg}")
        aug_str = f"_{'_'.join(aug_parts)}" if aug_parts else ""
        adapter_str = f"_{args.adapter}" if args.adapter and args.adapter != "none" else ""
        viz_dir = model_dir / "eval_results" / f"viz_{args.split}{aug_str}{adapter_str}_{timestamp}"
        viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating visualizations to: {viz_dir}")

        # Track z-slices used by CrossFeat for each case (to sync baseline viz)
        crossfeat_viz_slices: Dict[str, int] = {}

        if viz_enabled and viz_cache:
            for case_id, viz_data in viz_cache.items():
                print(f"  Visualizing {_display_id(case_id)}...")

                # Basic multi-view matches
                if args.viz or args.viz_compare_cross or args.viz_compare_tta:
                    z_slice_used = plot_matches_multiview(
                        viz_data,
                        viz_dir / f"{case_id}_matches.png"
                    )
                    if z_slice_used is not None:
                        crossfeat_viz_slices[case_id] = z_slice_used

                    # TRE distribution
                    if len(viz_data.tre_values) > 0:
                        plot_tre_distribution(
                            viz_data,
                            viz_dir / f"{case_id}_tre.png",
                            threshold=args.inlier_threshold
                        )

                # Crossing comparison
                if args.viz_compare_cross and case_id in viz_cache_no_cross:
                    plot_crossing_comparison(
                        viz_cache_no_cross[case_id],
                        viz_data,
                        viz_dir / f"{case_id}_comparison_cross.png"
                    )

                # TTA comparison
                if args.viz_compare_tta and case_id in viz_cache_no_tta:
                    tta_name = args.adapter.upper() if args.adapter else "TTA"
                    plot_tta_comparison(
                        viz_cache_no_tta[case_id],
                        viz_data,
                        tta_name,
                        viz_dir / f"{case_id}_comparison_tta.png"
                    )

            # Create summary figure
            result_dicts = [
                {"case_id": r.case_id, "top1_acc": r.top1_acc, "inlier_ratio": r.inlier_ratio, "tre_mean": r.tre_mean}
                for r in valid_results
            ]
            create_summary_figure(
                result_dicts,
                viz_dir / "summary.png",
                title=f"Evaluation Summary - {args.split} set"
            )

        if viz_baseline_enabled and viz_cache_baseline:
            baseline_dir = viz_dir / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Visualizing baseline to: {baseline_dir}")

            for case_id, viz_data in viz_cache_baseline.items():
                print(f"  Visualizing {_display_id(case_id)} baseline...")
                # Use the same z-slice as CrossFeat for fair comparison
                fixed_slice = crossfeat_viz_slices.get(case_id)
                plot_matches_multiview(
                    viz_data,
                    baseline_dir / f"{case_id}_matches.png",
                    fixed_slice_z=fixed_slice,
                )
                if len(viz_data.tre_values) > 0:
                    plot_tre_distribution(
                        viz_data,
                        baseline_dir / f"{case_id}_tre.png",
                        threshold=args.inlier_threshold
                    )

            baseline_result_dicts = [
                {"case_id": r.case_id, "top1_acc": r.top1_acc, "inlier_ratio": r.inlier_ratio, "tre_mean": r.tre_mean}
                for r in valid_baseline_results
            ]
            create_summary_figure(
                baseline_result_dicts,
                baseline_dir / "summary.png",
                title=f"Baseline Summary ({args.baseline}) - {args.split} set"
            )

        print(f"  Visualizations saved to: {viz_dir}")

    # Save results if output path provided
    if args.output or valid_results:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model_dir": str(model_dir),
            "dataset": dataset_name,
            "data_root": str(data_root),
            "split": args.split,
            "rotate_deg": args.rotate_deg,
            "translate_mm": args.translate_mm,
            "affine_scale": args.affine_scale,
            "affine_shear_deg": args.affine_shear_deg,
            "eval_config": {
                "model_file": f"{args.model}.pt",
                "use_full": args.use_full,
                "n_samples": args.n_samples,
                "inlier_threshold": args.inlier_threshold,
                "device": args.device,
                "seed": args.seed,
                "two_d_only": args.two_d_only,
                "match_per_slice": args.match_per_slice,
                "affine_scale": args.affine_scale,
                "affine_shear_deg": args.affine_shear_deg,
                "adapter": args.adapter if args.adapter and args.adapter != "none" else None,
                "adapter_config": adapter.get_config() if adapter is not None else None,
                "ratio_thresh": args.ratio_thresh,
                "baseline": args.baseline if args.baseline != "none" else None,
                "baseline_config": (
                    baseline_runner.config_dict()
                    if baseline_runner is not None
                    else ({"name": args.baseline, "error": baseline_error} if args.baseline != "none" else None)
                ),
            },
            "config": config,
            "summary": summary,
            "results": [asdict(r) for r in results],
            "baseline_summary": baseline_summary,
            "baseline_results": [asdict(r) for r in baseline_results] if baseline_results else [],
        }

        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = model_dir / "eval_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            aug_parts = []
            if args.rotate_deg > 0:
                aug_parts.append(f"rot{args.rotate_deg}")
            if args.translate_mm > 0:
                aug_parts.append(f"trans{args.translate_mm}")
            if args.affine_scale > 0:
                aug_parts.append(f"scale{args.affine_scale}")
            if args.affine_shear_deg > 0:
                aug_parts.append(f"shear{args.affine_shear_deg}")
            aug_str = f"_{'_'.join(aug_parts)}" if aug_parts else ""
            adapter_str = f"_{args.adapter}" if args.adapter and args.adapter != "none" else ""
            output_path = output_dir / f"eval_{args.split}{aug_str}{adapter_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    raise SystemExit("Use `python evaluate.py --help` for the public evaluation CLI.")
