#!/usr/bin/env python3
"""
CrossFeat Universal Training - Multi-Pair Crossing Model

Trains a single crossing model on multiple modality pairs simultaneously.
Supports both single-dataset (legacy) and multi-dataset configurations.

Usage:
    # Single dataset
    python train_universal.py --config /path/to/single_dataset.yaml

    # Multi-dataset
    python train_universal.py --config /path/to/multi_dataset.yaml
"""

import argparse
import gc
import json
import math
import pickle
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA

# Data loading (manifest-based)
from config.defaults import DATASET_ROOTS
from src.io.dataset_registry import DatasetManifest, build_manifest
from src.io.case_adapter import load_cases_from_manifest

# Feature extraction and models
from src.descriptor import SIFTDescriptor
from src.cross.models import DisentangledVAECrosser
from src.cross.losses import DisentangledVAELoss

# Training utilities
from src.extraction import extract_paired_descriptors, extract_paired_descriptors_per_case
from src.eval.metrics import compute_retrieval_accuracy
from src.utils.logging import setup_logging


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _parse_dataset_configs(config: dict) -> Tuple[List[dict], bool]:
    """Parse config into per-dataset configs, handling legacy and multi-dataset.

    Legacy config has top-level keys: data_dir, modalities, pairs, split_config.
    Multi-dataset config has a ``datasets`` list::

        datasets:
          - name: remind
            modalities: [t2, cet1]
            pairs: [{source: t2, target: cet1, use_crop: false}]
          - name: deliver
            modalities: [img, event]
            pairs: [{source: img, target: event}]

    Returns:
        (dataset_configs, is_multi_dataset)
        Each dataset_config dict has keys:
            name, modalities, pairs, data_root, manifest_kwargs.
    """
    if "datasets" in config:
        dataset_configs: List[dict] = []
        for ds in config["datasets"]:
            ds_name = ds["name"]
            ds_modalities = ds["modalities"]
            ds_pairs = ds.get("pairs", [
                {"source": a, "target": b}
                for a, b in combinations(ds_modalities, 2)
            ])
            ds_root = ds.get("data_root", DATASET_ROOTS.get(ds_name, ""))

            # For ReMIND, collect distinct use_crop values across pairs.
            # Different pairs may need different crop modes (e.g., MRI-MRI
            # full-volume vs MRI-US cropped).  We build one manifest per
            # distinct crop mode so each pair trains on the correct volumes.
            crop_modes: set = set()
            if ds_name == "remind":
                has_unannotated = False
                for p in ds_pairs:
                    if "use_crop" in p:
                        crop_modes.add(bool(p["use_crop"]))
                    else:
                        has_unannotated = True
                if not crop_modes:
                    # No pair has explicit use_crop: auto-detect from US presence
                    has_us = any(
                        "us" in (p.get("source", ""), p.get("target", ""))
                        for p in ds_pairs
                    )
                    crop_modes.add(has_us)
                elif has_unannotated:
                    # Some pairs annotated, some not: unannotated default to False
                    # (full volumes) to avoid silently cropping MRI-MRI pairs.
                    crop_modes.add(False)

            # Collect extra keys as manifest kwargs (e.g. stride, condition)
            # "canonical" is handled separately by _build_global_modality_index
            _RESERVED_DS_KEYS = {"name", "modalities", "pairs", "data_root", "canonical", "preprocess"}
            extra_kwargs = {
                k: v for k, v in ds.items() if k not in _RESERVED_DS_KEYS
            }
            canonical = ds.get("canonical", {})

            ds_preprocess_val = ds.get("preprocess")

            if len(crop_modes) <= 1:
                # Single crop mode (or non-remind dataset): one config entry
                manifest_kwargs: dict = dict(extra_kwargs)
                if crop_modes:
                    manifest_kwargs["use_crop"] = next(iter(crop_modes))
                entry: dict = {
                    "name": ds_name,
                    "modalities": ds_modalities,
                    "pairs": ds_pairs,
                    "data_root": ds_root,
                    "manifest_kwargs": manifest_kwargs,
                }
                if canonical:
                    entry["canonical"] = canonical
                if ds_preprocess_val is not None:
                    entry["preprocess"] = ds_preprocess_val
                dataset_configs.append(entry)
            else:
                # Multiple crop modes: split into separate config entries
                # per crop mode so each gets its own manifest.
                # Pairs without explicit use_crop default to False (no crop)
                # to avoid being assigned to both groups.
                for crop_val in sorted(crop_modes):
                    mode_pairs = []
                    for p in ds_pairs:
                        if bool(p.get("use_crop", False)) == crop_val:
                            # Annotate with effective use_crop so evaluation
                            # can determine the correct mode for each pair.
                            ap = p if "use_crop" in p else {**p, "use_crop": crop_val}
                            mode_pairs.append(ap)
                    if mode_pairs:
                        entry_crop: dict = {
                            "name": ds_name,
                            "modalities": ds_modalities,
                            "pairs": mode_pairs,
                            "data_root": ds_root,
                            "manifest_kwargs": {**extra_kwargs, "use_crop": crop_val},
                        }
                        if canonical:
                            entry_crop["canonical"] = canonical
                        if ds_preprocess_val is not None:
                            entry_crop["preprocess"] = ds_preprocess_val
                        dataset_configs.append(entry_crop)
        return dataset_configs, True

    # Legacy single-dataset mode
    dataset_name = config.get("dataset", "remind")
    modalities: list = config["modalities"]
    pairs = config.get("pairs", [
        {"source": a, "target": b}
        for a, b in combinations(modalities, 2)
    ])
    data_root = config.get("data_dir", DATASET_ROOTS.get(dataset_name, ""))

    # For ReMIND, handle per-pair crop modes
    crop_modes_legacy: set = set()
    if dataset_name == "remind":
        for p in pairs:
            if "use_crop" in p:
                crop_modes_legacy.add(bool(p["use_crop"]))
        if not crop_modes_legacy:
            has_us = any(
                "us" in (p.get("source", ""), p.get("target", ""))
                for p in pairs
            )
            crop_modes_legacy.add(has_us)

    if len(crop_modes_legacy) <= 1:
        kwargs: dict = {}
        if crop_modes_legacy:
            kwargs["use_crop"] = next(iter(crop_modes_legacy))
        return [{
            "name": dataset_name,
            "modalities": modalities,
            "pairs": pairs,
            "data_root": data_root,
            "manifest_kwargs": kwargs,
        }], False
    else:
        # Multiple crop modes: split into separate entries.
        # Pairs without explicit use_crop default to False (no crop).
        result_configs: List[dict] = []
        for crop_val in sorted(crop_modes_legacy):
            mode_pairs = []
            for p in pairs:
                if bool(p.get("use_crop", False)) == crop_val:
                    ap = p if "use_crop" in p else {**p, "use_crop": crop_val}
                    mode_pairs.append(ap)
            if mode_pairs:
                result_configs.append({
                    "name": dataset_name,
                    "modalities": modalities,
                    "pairs": mode_pairs,
                    "data_root": data_root,
                    "manifest_kwargs": {"use_crop": crop_val},
                })
        return result_configs, False


def _build_global_modality_index(
    dataset_configs: List[dict],
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], str]]:
    """Build modality index across datasets, with optional canonical merging.

    When a dataset config contains a ``canonical`` mapping, its modalities
    use the canonical name as the index key (shared across datasets).
    Without ``canonical``, keys are namespaced as ``dataset.modality``.

    Returns:
        (modality_to_idx, canonical_map) where:
        - modality_to_idx: canonical/namespaced key -> integer index
        - canonical_map: (dataset_name, native_mod) -> key used in modality_to_idx
    """
    index: Dict[str, int] = {}
    canonical_map: Dict[Tuple[str, str], str] = {}
    for ds_config in dataset_configs:
        ds_name = ds_config["name"]
        canonical = ds_config.get("canonical", {})
        for mod in ds_config["modalities"]:
            key = canonical.get(mod, f"{ds_name}.{mod}")
            if key not in index:
                index[key] = len(index)
            canonical_map[(ds_name, mod)] = key
    return index, canonical_map


def train_epoch_multi_pair(
    model: nn.Module,
    pair_datasets: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    loss_fn: DisentangledVAELoss,
    optimizer: optim.Optimizer,
    device: str,
    batch_size: int = 256,
    pair_sampling_alpha: float = 0.0,
    pair_local_props: Optional[Dict[Tuple[int, int], dict]] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """Train one epoch with multiple pairs.

    Args:
        model: Crosser model with num_modalities > 2
        pair_datasets: Dict mapping (mod_a_idx, mod_b_idx) to (d_a, d_b) arrays
        loss_fn: DisentangledVAELoss
        optimizer: Optimizer
        device: Device
        batch_size: Total batch size
        pair_sampling_alpha: Temperature for pair sampling. 0.0 = uniform (balanced),
            0.5 = square-root proportional, 1.0 = fully proportional to dataset size.
        pair_local_props: Optional dict mapping (mod_a_idx, mod_b_idx) to local
            property dicts with keys like "grad_a", "grad_b", "intensity_a", etc.
        epoch: Current epoch number (1-indexed, for VAE aux head warmup)

    Returns:
        Dict with 'loss', 'top1'
    """
    model.train()

    pairs = list(pair_datasets.keys())

    # Compute samples per pair: w_i = N_i^alpha, then normalize to batch_size
    pair_sizes = {p: len(pair_datasets[p][0]) for p in pairs}
    weights = {p: pair_sizes[p] ** pair_sampling_alpha for p in pairs}
    total_weight = sum(weights.values())
    samples_per_pair = {
        p: max(1, int(batch_size * weights[p] / total_weight))
        for p in pairs
    }

    # Determine epoch size (steps)
    max_samples = max(len(pair_datasets[p][0]) for p in pairs)
    steps_per_epoch = max(1, (max_samples + batch_size - 1) // batch_size)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step in range(steps_per_epoch):
        optimizer.zero_grad()

        # Collect batch from all pairs
        batch_a_list: list = []
        batch_b_list: list = []
        mod_a_list: List[int] = []
        mod_b_list: List[int] = []

        # Local props accumulators (only used when pair_local_props is set)
        grad_a_list: list = []
        grad_b_list: list = []
        intensity_a_list: list = []
        intensity_b_list: list = []
        coords_a_list: list = []
        coords_b_list: list = []

        for (a_idx, b_idx) in pairs:
            d_a, d_b = pair_datasets[(a_idx, b_idx)]
            n = len(d_a)

            # Random sample from this pair
            take = min(samples_per_pair[(a_idx, b_idx)], n)

            idx = np.random.choice(n, size=take, replace=(take > n))

            batch_a_list.append(d_a[idx])
            batch_b_list.append(d_b[idx])
            mod_a_list.extend([a_idx] * take)
            mod_b_list.extend([b_idx] * take)

            # Collect local props with same idx for row-level alignment
            if pair_local_props and (a_idx, b_idx) in pair_local_props:
                lp = pair_local_props[(a_idx, b_idx)]
                if "grad_a" in lp:
                    grad_a_list.append(lp["grad_a"][idx])
                    grad_b_list.append(lp["grad_b"][idx])
                if "intensity_a" in lp:
                    intensity_a_list.append(lp["intensity_a"][idx])
                    intensity_b_list.append(lp["intensity_b"][idx])
                if "coords_a" in lp:
                    coords_a_list.append(lp["coords_a"][idx])
                    coords_b_list.append(lp["coords_b"][idx])

        # Stack into tensors
        batch_a = torch.tensor(np.vstack(batch_a_list), dtype=torch.float32, device=device)
        batch_b = torch.tensor(np.vstack(batch_b_list), dtype=torch.float32, device=device)
        mod_a = torch.tensor(mod_a_list, dtype=torch.long, device=device)
        mod_b = torch.tensor(mod_b_list, dtype=torch.long, device=device)

        # VAE loss takes (model, d_a, d_b, mod_a, mod_b, **kwargs)
        vae_kwargs: dict = {}
        if grad_a_list and getattr(loss_fn, "supports_probe_c_labels", False):
            vae_kwargs["grad_magnitudes_a"] = torch.tensor(
                np.concatenate(grad_a_list), dtype=torch.float32, device=device,
            )
            vae_kwargs["grad_magnitudes_b"] = torch.tensor(
                np.concatenate(grad_b_list), dtype=torch.float32, device=device,
            )
        if intensity_a_list and getattr(loss_fn, "supports_probe_bc_labels", False):
            vae_kwargs["intensities_a"] = torch.tensor(
                np.concatenate(intensity_a_list), dtype=torch.float32, device=device,
            )
            vae_kwargs["intensities_b"] = torch.tensor(
                np.concatenate(intensity_b_list), dtype=torch.float32, device=device,
            )
        if coords_a_list and hasattr(loss_fn, "lambda_coord_head"):
            vae_kwargs["coords_a"] = torch.tensor(
                np.concatenate(coords_a_list), dtype=torch.float32, device=device,
            )
            vae_kwargs["coords_b"] = torch.tensor(
                np.concatenate(coords_b_list), dtype=torch.float32, device=device,
            )
        if hasattr(loss_fn, "aux_warmup_epochs"):
            vae_kwargs["epoch"] = epoch

        loss = loss_fn(model, batch_a, batch_b, mod_a, mod_b, **vae_kwargs)

        # Reuse crossed prediction stashed by the loss to avoid redundant forward
        stashed = getattr(loss_fn, "_last_d_a_crossed", None)
        if stashed is not None:
            pred = stashed
        else:
            with torch.no_grad():
                pred = model(batch_a, mod_a, mod_b, normalize=True)
        target = batch_b / (batch_b.norm(dim=1, keepdim=True) + 1e-8)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute top-1 accuracy
        with torch.no_grad():
            sim = pred @ target.T
            correct = (sim.argmax(dim=1) == torch.arange(len(pred), device=device)).sum().item()
            total_correct += correct
            total_samples += len(pred)

    return {
        "loss": total_loss / steps_per_epoch,
        "top1": 100.0 * total_correct / max(1, total_samples),
    }


def evaluate_pair(
    model: nn.Module,
    cases_with_mods: List[tuple],
    extractor: object,
    pca: PCA,
    device: str,
    mod_a_idx: int,
    mod_b_idx: int,
    n_samples: int = 1000,
    seed: int = 42,
    pair_sampling: str = "random",
    max_kpts_per_slice: int = 500,
    num_workers: int = 1,
    precomputed_per_case: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, float]:
    """Evaluate model on a specific pair.

    Args:
        cases_with_mods: List of (CaseData, mod_a_name, mod_b_name) tuples.
    """
    if not cases_with_mods:
        return {"top1": 0.0, "top5": 0.0, "n_cases": 0}

    # Switch to eval mode (disables dropout)
    was_training = model.training
    model.eval()

    # Extract all cases and get modality names from first case
    cases_only = [c[0] for c in cases_with_mods]
    mod_a_name = cases_with_mods[0][1]
    mod_b_name = cases_with_mods[0][2]

    per_case_descs: List[Tuple[np.ndarray, np.ndarray]] = []
    if precomputed_per_case is not None:
        per_case_descs = precomputed_per_case
    else:
        per_case_results = extract_paired_descriptors_per_case(
            cases_only,
            [mod_a_name, mod_b_name],
            extractor,
            n_samples_per_case=n_samples,
            seed=seed,
            pair_sampling=pair_sampling,
            max_kpts_per_slice=max_kpts_per_slice,
            show_progress=False,  # Quiet during validation
            num_workers=num_workers,
        )
        for desc, n in per_case_results:
            if n == 0:
                continue
            d_a = pca.transform(desc[mod_a_name]).astype(np.float32)
            d_b = pca.transform(desc[mod_b_name]).astype(np.float32)
            per_case_descs.append((d_a, d_b))

    all_top1: List[float] = []
    all_top5: List[float] = []

    for d_a, d_b in per_case_descs:
        # Cross descriptors
        with torch.inference_mode():
            d_a_t = torch.tensor(d_a, dtype=torch.float32, device=device)
            mod_a_t = torch.full((len(d_a),), mod_a_idx, dtype=torch.long, device=device)
            mod_b_t = torch.full((len(d_a),), mod_b_idx, dtype=torch.long, device=device)

            pred = model(d_a_t, mod_a_t, mod_b_t, normalize=True).cpu().numpy()

        # Compute metrics
        d_b_norm = d_b / (np.linalg.norm(d_b, axis=1, keepdims=True) + 1e-8)
        metrics = compute_retrieval_accuracy(pred, d_b_norm)
        all_top1.append(metrics["top1"])
        all_top5.append(metrics["top5"])

    # Restore training mode if it was active
    if was_training:
        model.train()

    if not all_top1:
        return {"top1": 0.0, "top5": 0.0, "n_cases": 0}

    return {
        "top1": float(np.mean(all_top1)),
        "top5": float(np.mean(all_top5)),
        "n_cases": len(all_top1),
    }


def _create_vae_loss(config: dict, num_modalities: int, device: str) -> DisentangledVAELoss:
    """Create DisentangledVAELoss from config dict."""
    import logging
    _logger = logging.getLogger(__name__)

    vae_adv_weight = config.get("vae_adversarial", 0.0)
    if vae_adv_weight > 0 and not config.get("adversarial", False):
        _logger.warning(
            "vae_adversarial=%.4f but adversarial is not enabled in config. "
            "Adversarial weight will be set to 0. Set adversarial: true to enable.",
            vae_adv_weight,
        )

    return DisentangledVAELoss(
        lambda_recon=config.get("vae_recon", 1.0),
        lambda_geom_consistency=config.get("vae_geom", 0.5),
        lambda_crossing=config.get("vae_crossing", 1.0),
        lambda_diversity=config.get("vae_diversity", 0.1),
        lambda_contrastive=config.get("vae_contrastive", 0.5),
        temperature=config.get("temperature", 0.07),
        min_geom_norm=config.get("min_geom_norm", 0.0),
        lambda_adversarial=(
            config.get("vae_adversarial", 0.0)
            if config.get("adversarial", False)
            else 0.0
        ),
        lambda_geometry=config.get("geo", 0.0),
        geometry_type=config.get("geo_type", "similarity"),
        z_app_dropout=config.get("z_app_dropout", 0.0),
        lambda_latent_contrastive=config.get("vae_latent_contrastive", 0.0),
        lambda_latent_leakage=config.get("vae_latent_leakage", 0.0),
        latent_leakage_type=config.get("vae_leakage_type", "cross_cov"),
        lambda_app_repulsion=config.get("vae_app_repulsion", 0.0),
        lambda_app_mod_supcon=config.get("vae_app_mod_supcon", 0.0),
        app_mod_supcon_temperature=config.get("vae_app_mod_supcon_temp", 0.07),
        lambda_t4_margin=config.get("vae_t4_margin", 0.0),
        t4_margin=config.get("vae_t4_margin_target", 0.10),
        lambda_t1_margin=config.get("vae_t1_margin", 0.0),
        t1_margin_target=config.get("vae_t1_margin_target", 2.0),
        t1_margin_min_delta=config.get("vae_t1_margin_min_delta", 0.05),
        lambda_probe_c_margin=config.get("vae_probe_c_margin", 0.0),
        probe_c_target_gap=config.get("vae_probe_c_target", 0.05),
        probe_c_ridge=config.get("vae_probe_c_ridge", 1e-3),
        probe_c_invert=config.get("vae_probe_c_invert", False),
        lambda_probe_bc_margin=config.get("vae_probe_bc_margin", 0.0),
        probe_bc_target_b=config.get("vae_probe_bc_target_b", 0.05),
        probe_bc_target_c=config.get("vae_probe_bc_target_c", 0.05),
        lambda_kl_geom=config.get("vae_kl_geom", 0.0),
        lambda_kl_app=config.get("vae_kl_app", 0.0),
        num_modalities=num_modalities,
        d_app=config.get("d_app", 64),
        free_bits=config.get("vae_free_bits", 0.0),
        geom_reg_scale=config.get("vae_geom_reg_scale", 1.0),
        lambda_grad_head=config.get("vae_grad_head", 0.0),
        lambda_modality_heads=config.get("vae_modality_heads", 0.0),
        lambda_coord_head=config.get("vae_coord_head", 0.0),
        aux_warmup_epochs=config.get("vae_aux_warmup_epochs", 10),
        hard_negative_ratio=config.get("vae_hard_negative_ratio", 0.0),
        hard_negative_weight=config.get("vae_hard_negative_weight", 2.0),
        symmetric_infonce=config.get("vae_symmetric_infonce", False),
        stop_grad_z_geom_recon=config.get("vae_stop_grad_z_geom_recon", False),
        lambda_vicreg_cov=config.get("vae_vicreg_cov", 0.0),
        descriptor_noise_std=config.get("vae_descriptor_noise_std", 0.0),
    ).to(device)


def _prepare_output_dir(config: dict) -> Path:
    """Create a new run directory without overwriting an existing bundle."""
    output_dir = (
        Path(config.get("output_dir", "experiments"))
        / config.get("exp", "universal")
        / config.get("name", "universal")
    )
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir()}
        # A failure before training may leave only diagnostics. Those are safe
        # to reuse; any model/unknown artifact must be preserved under its run.
        protected = existing - {"logs", "config_source.yaml"}
        if protected:
            raise FileExistsError(
                f"Output directory contains existing artifacts: {output_dir} "
                f"({', '.join(sorted(protected))}). Choose a new config 'name' "
                "to preserve the existing run."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Multi-Pair CrossFeat Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup output directory
    output_dir = _prepare_output_dir(config)

    logger = setup_logging(output_dir)

    # Set seeds for reproducibility (including CUDA)
    import random
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("=" * 70)
    logger.info("Universal CrossFeat Training")
    logger.info("=" * 70)

    # Save config
    with open(output_dir / "config_source.yaml", "w") as f:
        import yaml
        yaml.dump(config, f)

    # ---------- Parse dataset configs ----------
    dataset_configs, is_multi_dataset = _parse_dataset_configs(config)

    if is_multi_dataset:
        logger.info("Multi-dataset mode: %d datasets", len(dataset_configs))
        for ds in dataset_configs:
            logger.info(
                "  %s: modalities=%s, %d pairs",
                ds["name"], ds["modalities"], len(ds["pairs"]),
            )
    else:
        logger.info("Single-dataset mode: %s", dataset_configs[0]["name"])

    # ---------- Build modality index ----------
    if is_multi_dataset:
        modality_to_idx, canonical_map = _build_global_modality_index(dataset_configs)
    else:
        mod_list = dataset_configs[0]["modalities"]
        modality_to_idx = {m: i for i, m in enumerate(mod_list)}
        canonical_map = {}

    num_modalities = len(modality_to_idx)
    logger.info("Modality index (%d): %s", num_modalities, modality_to_idx)

    # Build lookup from namespaced key to integer index.
    # When canonical mapping is active, multiple namespaced keys may map
    # to the same canonical index (e.g., deliver.img and qxs_saropt.opt both -> rgb).
    if is_multi_dataset:
        key_to_idx: Dict[str, int] = {}
        for (ds_name, mod), canon_key in canonical_map.items():
            ns_key = f"{ds_name}.{mod}"
            key_to_idx[ns_key] = modality_to_idx[canon_key]
    else:
        key_to_idx = modality_to_idx  # no-op alias for single-dataset

    # Collect all directed pairs across datasets
    # Each entry: (qualified_a, qualified_b, dataset_name)
    all_pairs: List[Tuple[str, str, str]] = []
    for ds_config in dataset_configs:
        ds_name = ds_config["name"]
        for p in ds_config["pairs"]:
            if is_multi_dataset:
                a_key = f"{ds_name}.{p['source']}"
                b_key = f"{ds_name}.{p['target']}"
            else:
                a_key = p["source"]
                b_key = p["target"]
            all_pairs.append((a_key, b_key, ds_name))

    logger.info("Training pairs: %s", [(a, b) for a, b, _ in all_pairs])

    # ---------- Create extractor ----------
    extractor_kwargs: dict = {}
    descriptor = config.get("descriptor", "sift")
    if str(descriptor).lower() != "sift":
        raise ValueError(
            f"Unsupported descriptor={descriptor!r}. CrossFeat supports only descriptor='sift'."
        )
    if config.get("rootsift", False):
        extractor_kwargs["rootsift"] = True
    # Top-level preprocess serves as default for all datasets
    default_preprocess = config.get("preprocess", "none")
    if default_preprocess != "none":
        extractor_kwargs["preprocess"] = default_preprocess
    extractor = SIFTDescriptor(**extractor_kwargs)
    logger.info("Descriptor: %s", descriptor)

    # Per-dataset preprocessing overrides (e.g. gaussian_sobel for SAR only).
    # Maps dataset name -> preprocess mode; datasets without an entry use the
    # extractor's default (set above from top-level config or "none").
    ds_preprocess: dict[str, str] = {}
    for ds_cfg in dataset_configs:
        pp = ds_cfg.get("preprocess")
        if pp is not None:
            ds_preprocess[ds_cfg["name"]] = pp
    if ds_preprocess:
        logger.info("Per-dataset preprocessing: %s", ds_preprocess)

    # ---------- VAE-only mode ----------
    model_type = str(config.get("model", "vae")).lower()
    if model_type != "vae":
        raise ValueError(
            f"Unsupported model={model_type!r}. CrossFeat supports only model='vae'."
        )
    use_probe_c = config.get("vae_probe_c_margin", 0.0) > 0
    use_probe_bc = config.get("vae_probe_bc_margin", 0.0) > 0
    use_grad_head = config.get("vae_grad_head", 0.0) > 0
    use_coord_head = config.get("vae_coord_head", 0.0) > 0
    use_local_props = use_probe_c or use_probe_bc or use_grad_head or use_coord_head

    # ---------- Build manifests ----------
    # Build one manifest per (dataset, pair, crop_mode) so each pair only
    # requires its two modalities, maximizing available cases.  Previously
    # we built one manifest per dataset with ALL modalities, which meant
    # only cases with every modality were used (e.g. 6 instead of 40 for
    # ReMIND t2-cet1 because few cases have all of t2+cet1+us+flair).
    manifests: dict = {}
    pair_to_manifest: Dict[Tuple[str, str], DatasetManifest] = {}

    for ds_config in dataset_configs:
        ds_name = ds_config["name"]
        mk = ds_config.get("manifest_kwargs", {})

        for p in ds_config["pairs"]:
            pair_mods = [p["source"], p["target"]]
            # Key includes pair modalities so each pair gets its own manifest
            mk_key = (ds_name, tuple(pair_mods), tuple(sorted(mk.items())))

            if mk_key not in manifests:
                logger.info(
                    "Building manifest for %s %s->%s (kwargs=%s)...",
                    ds_name, p["source"], p["target"], mk,
                )
                manifest = build_manifest(
                    ds_name,
                    ds_config["data_root"],
                    pair_mods,
                    **mk,
                )
                manifests[mk_key] = manifest
                logger.info(
                    "  %d train, %d val records",
                    len(manifest.train_records),
                    len(manifest.val_records),
                )

            if is_multi_dataset:
                a_key = f"{ds_name}.{p['source']}"
                b_key = f"{ds_name}.{p['target']}"
            else:
                a_key = p["source"]
                b_key = p["target"]
            pair_to_manifest[(a_key, b_key)] = manifests[mk_key]

    # ---------- uniform_pair_data: compute extraction order ----------
    uniform_pair_data = config.get("uniform_pair_data", False)
    uniform_target_samples: int | None = None

    if uniform_pair_data:
        # Estimate descriptor yield per pair from manifest record counts.
        # For NIfTI: group_records_by_key gives case count; for 2D: one record = one case.
        spc = config.get("samples_per_case", 2048)
        max_kpts = config.get("max_kpts_per_slice", 500)
        pair_sampling = config.get("pair_sampling", "random")
        pair_estimates: dict[tuple[str, str], int] = {}
        for a_key, b_key, _ds in all_pairs:
            manifest = pair_to_manifest[(a_key, b_key)]
            train_recs = manifest.train_records
            is_nifti = bool(train_recs and train_recs[0].slice_idx is not None)
            if is_nifti:
                # NIfTI: multiple records per case (slices), group by case
                groups = manifest.group_records_by_key("train")
                n_cases = len(groups)
                per_case = spc
            else:
                # 2D: each record is one case (image)
                n_cases = len(train_recs)
                if pair_sampling in {"2d_sift_ref", "2d_sift_dual"}:
                    per_case = min(spc, max_kpts)
                else:
                    per_case = spc
            pair_estimates[(a_key, b_key)] = n_cases * per_case
        # Sort ascending so smallest pair is extracted first
        all_pairs_sorted = sorted(
            all_pairs, key=lambda t: pair_estimates[(t[0], t[1])]
        )
        logger.info(
            "uniform_pair_data: extraction order (smallest first): %s",
            [(f"{a}->{b}", pair_estimates[(a, b)]) for a, b, _ in all_pairs_sorted],
        )
    else:
        all_pairs_sorted = list(all_pairs)

    # ---------- Extract descriptors per pair ----------
    pair_data: dict = {}
    # Cache loaded cases per (manifest_id, split) to avoid redundant loading.
    # Using id(manifest) ensures different crop modes get different caches.
    cases_cache: dict = {}

    for a_key, b_key, ds_name in all_pairs_sorted:
        manifest = pair_to_manifest[(a_key, b_key)]

        # Get clean modality names (strip namespace if present)
        mod_a = a_key.split(".", 1)[1] if is_multi_dataset else a_key
        mod_b = b_key.split(".", 1)[1] if is_multi_dataset else b_key

        logger.info("Loading %s->%s (dataset=%s)", mod_a, mod_b, ds_name)

        # Load cases from manifest (cached per manifest instance + split)
        max_train_cases = config.get("max_train_cases")
        for split_name in ("train", "val"):
            cache_key = (id(manifest), split_name)
            if cache_key not in cases_cache:
                if split_name == "train":
                    mc = max_train_cases
                elif max_train_cases is not None:
                    mc = max(1, max_train_cases // 5)
                else:
                    mc = None
                cases_cache[cache_key] = load_cases_from_manifest(
                    manifest, split_name, max_cases=mc,
                )
        train_cases = cases_cache[(id(manifest), "train")]
        val_cases = cases_cache[(id(manifest), "val")]

        if not train_cases:
            logger.warning("No training cases for %s->%s, skipping", mod_a, mod_b)
            continue

        logger.info("  %d train, %d val cases", len(train_cases), len(val_cases))

        # Apply per-dataset preprocessing override (e.g. gaussian_sobel for SAR)
        if ds_name in ds_preprocess:
            extractor.preprocess = ds_preprocess[ds_name]
        elif hasattr(extractor, "preprocess"):
            extractor.preprocess = default_preprocess

        # Extract training descriptors
        result = extract_paired_descriptors(
            train_cases,
            [mod_a, mod_b],
            extractor,
            n_samples_per_case=config.get("samples_per_case", 2048),
            seed=seed,
            pair_sampling=config.get("pair_sampling", "random"),
            max_kpts_per_slice=config.get("max_kpts_per_slice", 500),
            num_workers=config.get("extraction_workers", 1),
            return_local_props=use_local_props,
            max_samples=uniform_target_samples,
        )
        if use_local_props:
            descs, n_samples, local_props = result
            # Validate that requested local props were actually extracted
            if (use_probe_c or use_grad_head) and "gradient" not in local_props:
                raise ValueError(
                    f"Config requests gradient-based losses but extraction for "
                    f"{mod_a}->{mod_b} returned no gradient data."
                )
            if use_probe_bc and "intensity" not in local_props:
                raise ValueError(
                    f"Config requests intensity-based losses but extraction for "
                    f"{mod_a}->{mod_b} returned no intensity data."
                )
            if use_coord_head and "coords" not in local_props:
                raise ValueError(
                    f"Config requests coord_head loss but extraction for "
                    f"{mod_a}->{mod_b} returned no coords data."
                )
        else:
            descs, n_samples = result
            local_props = None

        if n_samples == 0:
            logger.warning("No samples extracted for %s->%s, skipping", mod_a, mod_b)
            continue

        logger.info("  Extracted %d training samples", n_samples)

        # Update uniform cap: initialize from first pair, then ratchet
        # down if any subsequent pair yields fewer than the current cap.
        if uniform_pair_data:
            if uniform_target_samples is None:
                uniform_target_samples = n_samples
                logger.info(
                    "uniform_pair_data: initial cap %d (from %s->%s)",
                    uniform_target_samples, mod_a, mod_b,
                )
            elif n_samples < uniform_target_samples:
                logger.info(
                    "uniform_pair_data: cap lowered %d -> %d (from %s->%s)",
                    uniform_target_samples, n_samples, mod_a, mod_b,
                )
                uniform_target_samples = n_samples

        # Build val_cases_with_mods for evaluate_pair compatibility
        val_cases_with_mods = [(c, mod_a, mod_b) for c in val_cases]

        # Display key: include dataset name in multi-dataset mode
        if is_multi_dataset:
            display = f"{ds_name}:{mod_a}->{mod_b}"
        else:
            display = f"{mod_a}->{mod_b}"

        pair_data[(a_key, b_key)] = {
            "val_cases_with_mods": val_cases_with_mods,
            "descs_raw": {mod_a: descs[mod_a], mod_b: descs[mod_b]},
            "mod_a": mod_a,
            "mod_b": mod_b,
            "ds_name": ds_name,
            "display": display,
            "local_props": local_props,
        }

        # Free train volumes immediately — descriptors are already extracted.
        # Each pair has its own manifest so no other pair needs these cases.
        # Val cases remain alive via val_cases_with_mods references in pair_data.
        train_cache_key = (id(manifest), "train")
        if train_cache_key in cases_cache:
            del cases_cache[train_cache_key]
            gc.collect()

    if not pair_data:
        logger.error("No valid pairs found!")
        raise RuntimeError(
            "No valid training pairs were found. Check dataset data_root, file layout, "
            "and split contents."
        )

    # ---------- Fit shared PCA on pooled descriptors ----------
    logger.info("Fitting shared PCA on pooled descriptors...")
    pca_dim = config.get("pca_dim", 128)

    pooled_parts: list = []
    for data in pair_data.values():
        for key in data["descs_raw"]:
            arr = data["descs_raw"][key]
            # Per-modality cap of 50K samples
            max_per_mod = 50000
            if len(arr) > max_per_mod:
                idx = np.random.choice(len(arr), size=max_per_mod, replace=False)
                pooled_parts.append(arr[idx])
            else:
                pooled_parts.append(arr)

    pooled = np.vstack(pooled_parts)
    logger.info("Pooled samples: %d, dim=%d", len(pooled), pooled.shape[1])

    pca = PCA(n_components=pca_dim, whiten=True)
    pca.fit(pooled)
    logger.info(
        "PCA: %d -> %d (%.1f%% var)",
        pooled.shape[1], pca_dim, sum(pca.explained_variance_ratio_) * 100,
    )
    del pooled, pooled_parts

    # ---------- Transform descriptors with PCA ----------
    pair_datasets: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    pair_local_props: Dict[Tuple[int, int], dict] = {}

    for (a_key, b_key), data in pair_data.items():
        mod_a = data["mod_a"]
        mod_b = data["mod_b"]

        d_a = pca.transform(data["descs_raw"][mod_a]).astype(np.float32)
        d_b = pca.transform(data["descs_raw"][mod_b]).astype(np.float32)

        a_idx = key_to_idx[a_key]
        b_idx = key_to_idx[b_key]

        # Both directions — concatenate if canonical mapping merges pairs
        for idx_pair, descs in [((a_idx, b_idx), (d_a, d_b)),
                                ((b_idx, a_idx), (d_b, d_a))]:
            if idx_pair in pair_datasets:
                existing_a, existing_b = pair_datasets[idx_pair]
                pair_datasets[idx_pair] = (
                    np.concatenate([existing_a, descs[0]]),
                    np.concatenate([existing_b, descs[1]]),
                )
            else:
                pair_datasets[idx_pair] = descs

        # Build local props for both directions (scalar labels — no PCA needed)
        lp = data.get("local_props")
        if lp is not None:
            fwd_lp: dict = {}
            rev_lp: dict = {}
            if "gradient" in lp:
                fwd_lp["grad_a"] = lp["gradient"][mod_a]
                fwd_lp["grad_b"] = lp["gradient"][mod_b]
                rev_lp["grad_a"] = lp["gradient"][mod_b]
                rev_lp["grad_b"] = lp["gradient"][mod_a]
            if "intensity" in lp:
                fwd_lp["intensity_a"] = lp["intensity"][mod_a]
                fwd_lp["intensity_b"] = lp["intensity"][mod_b]
                rev_lp["intensity_a"] = lp["intensity"][mod_b]
                rev_lp["intensity_b"] = lp["intensity"][mod_a]
            if "coords" in lp:
                # Coords are the same for both modalities (aligned data)
                fwd_lp["coords_a"] = lp["coords"][mod_a]
                fwd_lp["coords_b"] = lp["coords"][mod_b]
                rev_lp["coords_a"] = lp["coords"][mod_b]
                rev_lp["coords_b"] = lp["coords"][mod_a]
            for idx_pair, lp_dict in [((a_idx, b_idx), fwd_lp),
                                      ((b_idx, a_idx), rev_lp)]:
                if idx_pair in pair_local_props:
                    existing = pair_local_props[idx_pair]
                    for k in lp_dict:
                        if k in existing:
                            existing[k] = np.concatenate([existing[k], lp_dict[k]])
                        else:
                            existing[k] = lp_dict[k]
                else:
                    pair_local_props[idx_pair] = lp_dict

        # Free raw descriptors
        data["descs_raw"] = None

    # Free loaded image data — cases_cache holds all volumes/images from
    # all datasets and is no longer needed after PCA transform.
    del cases_cache
    gc.collect()

    # --- uniform_pair_data: final trim to exact target ---
    if uniform_pair_data and uniform_target_samples is not None:
        for pair_key in list(pair_datasets.keys()):
            d_a, d_b = pair_datasets[pair_key]
            if len(d_a) > uniform_target_samples:
                # Same seed per pair so forward (a,b) and reverse (b,a)
                # get identical indices, preserving descriptor pair alignment.
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(d_a), size=uniform_target_samples, replace=False)
                idx.sort()  # preserve ordering
                pair_datasets[pair_key] = (d_a[idx], d_b[idx])
                if pair_key in pair_local_props:
                    lp = pair_local_props[pair_key]
                    for k in list(lp.keys()):
                        lp[k] = lp[k][idx]
        sizes = {k: len(v[0]) for k, v in pair_datasets.items()}
        logger.info("uniform_pair_data: final pair sizes: %s", sizes)

    logger.info("Created %d directed pair datasets", len(pair_datasets))
    if pair_local_props:
        logger.info("Local props available for %d directed pairs", len(pair_local_props))

    # ---------- Create model ----------
    device = config.get("device", "cuda")
    model = DisentangledVAECrosser(
        descriptor_dim=pca_dim,
        num_modalities=num_modalities,
        hidden_dim=config.get("hidden_dim", 128),
        dropout=config.get("dropout", 0.15),
        d_geom=config.get("d_geom", 64),
        d_app=config.get("d_app", 64),
        embed_dim=config.get("embed_dim", 16),
        condition_decoder=config.get("condition_decoder", False),
        variational=config.get("variational", False),
        variational_app_only=config.get("variational_app_only", False),
        use_adversarial=config.get("adversarial", False),
        normalize_geom=config.get("normalize_geom", False),
        geom_scale=config.get("geom_scale", 1.0),
        simple_crosser=config.get("simple_crosser", False),
        separate_encoder=config.get("separate_encoder", False),
        modality_specific_encoder=config.get("modality_specific_encoder", False),
        fsq_levels=config.get("fsq_levels", 0),
        use_grad_head=use_grad_head,
        use_modality_heads=bool(config.get("vae_modality_heads", 0.0) > 0),
        use_coord_head=use_coord_head,
        crosser_hidden_dim=config.get("crosser_hidden_dim", None) or None,
        disentangle=config.get("disentangle", True),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %s, params: %s, modalities: %d",
        model_type, f"{n_params:,}", num_modalities,
    )

    # ---------- Create loss function ----------
    loss_fn = _create_vae_loss(config, num_modalities, device)
    logger.info("Created DisentangledVAELoss for VAE training")

    # Include loss_fn parameters (e.g., VAE's learnable priors) in optimizer.
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = optim.AdamW(
        all_params,
        lr=float(config.get("lr", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(config.get("epochs", 100)),
    )

    # ---------- Training loop ----------
    epochs = config.get("epochs", 100)
    patience = config.get("patience", 30)
    batch_size = config.get("batch_size", 256)
    # Temperature-scaled pair sampling: alpha=0 uniform, alpha=1 proportional
    if "pair_sampling_alpha" in config:
        pair_sampling_alpha = float(config["pair_sampling_alpha"])
    else:
        # Backward compat: pair_balanced_sampling=True → alpha=0, False → alpha=1
        pair_sampling_alpha = 0.0 if config.get("pair_balanced_sampling", True) else 1.0
    logger.info("Pair sampling alpha: %.2f (0=uniform, 1=proportional)", pair_sampling_alpha)

    best_val_mean_top1 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_val_by_pair: Dict[str, float] = {}

    # VAE hard-negative warmup
    warmup = config.get("warmup", 25)
    vae_hn_ratio = config.get("vae_hard_negative_ratio", 0.0)
    if warmup > 0 and vae_hn_ratio > 0:
        loss_fn.hard_negative_ratio = 0.0
        logger.info("VAE HNM warmup: disabled for first %d epochs", warmup)

    # KL warmup annealing for VAE: linearly ramp KL weights over N epochs
    kl_warmup_epochs = config.get("kl_warmup_epochs", 0)
    target_kl_geom = 0.0
    target_kl_app = 0.0
    if hasattr(loss_fn, "lambda_kl_geom"):
        target_kl_geom = loss_fn.lambda_kl_geom
        target_kl_app = loss_fn.lambda_kl_app
        if kl_warmup_epochs > 0 and (target_kl_geom > 0 or target_kl_app > 0):
            loss_fn.lambda_kl_geom = 0.0
            loss_fn.lambda_kl_app = 0.0
            logger.info("KL warmup: linearly ramping over %d epochs", kl_warmup_epochs)

    # Build validation info
    val_pairs_info: list = []
    for (a_key, b_key), data in pair_data.items():
        if data["val_cases_with_mods"]:
            val_pairs_info.append({
                "display": data["display"],
                "cases": data["val_cases_with_mods"],
                "mod_a_idx": key_to_idx[a_key],
                "mod_b_idx": key_to_idx[b_key],
                "ds_name": data["ds_name"],
            })

    val_n_samples = int(config.get("val_n_samples", 1000))
    val_seed = int(config.get("val_seed", 9999))
    val_extraction_workers = int(
        config.get("val_extraction_workers", config.get("extraction_workers", 1))
    )

    if val_pairs_info:
        logger.info(
            "Precomputing validation descriptors once "
            "(n_samples=%d, seed=%d, workers=%d)...",
            val_n_samples,
            val_seed,
            val_extraction_workers,
        )
        for info in val_pairs_info:
            # Apply per-dataset preprocessing override for val extraction
            val_ds = info["ds_name"]
            if val_ds in ds_preprocess:
                extractor.preprocess = ds_preprocess[val_ds]
            elif hasattr(extractor, "preprocess"):
                extractor.preprocess = default_preprocess

            cases_only = [c[0] for c in info["cases"]]
            mod_a_name = info["cases"][0][1]
            mod_b_name = info["cases"][0][2]
            per_case = extract_paired_descriptors_per_case(
                cases_only,
                [mod_a_name, mod_b_name],
                extractor,
                n_samples_per_case=val_n_samples,
                seed=val_seed,
                pair_sampling=config.get("pair_sampling", "random"),
                max_kpts_per_slice=config.get("max_kpts_per_slice", 500),
                show_progress=False,
                num_workers=val_extraction_workers,
            )
            cached_descs: List[Tuple[np.ndarray, np.ndarray]] = []
            for desc, n in per_case:
                if n == 0:
                    continue
                d_a = pca.transform(desc[mod_a_name]).astype(np.float32)
                d_b = pca.transform(desc[mod_b_name]).astype(np.float32)
                cached_descs.append((d_a, d_b))
            info["cached_descs"] = cached_descs
            logger.info(
                "  %s: cached %d/%d cases",
                info["display"],
                len(cached_descs),
                len(per_case),
            )

    # Free val case volumes — descriptors are precomputed in cached_descs.
    # Clear CaseData.volumes dicts to release image arrays while keeping
    # the CaseData shell (evaluate_pair needs cases_with_mods to be non-empty
    # and indexes [0] for modality names).
    _freed_ids: set = set()
    for info in val_pairs_info:
        for case, _, _ in info["cases"]:
            cid = id(case)
            if cid not in _freed_ids:
                case.volumes.clear()
                _freed_ids.add(cid)
    del _freed_ids
    gc.collect()
    logger.info("Freed all case volumes (train + val) after descriptor extraction")

    # Initial validation (epoch 0) to establish baseline
    if val_pairs_info:
        logger.info("Initial validation (epoch 0) to verify model initialization...")
        val_top1_by_pair: Dict[str, float] = {}
        for info in val_pairs_info:
            metrics = evaluate_pair(
                model, info["cases"], extractor, pca, device,
                info["mod_a_idx"], info["mod_b_idx"],
                n_samples=val_n_samples, seed=val_seed,
                pair_sampling=config.get("pair_sampling", "random"),
                max_kpts_per_slice=config.get("max_kpts_per_slice", 500),
                num_workers=val_extraction_workers,
                precomputed_per_case=info.get("cached_descs"),
            )
            val_top1_by_pair[info["display"]] = metrics["top1"]

        if val_top1_by_pair:
            mean_top1 = float(np.mean(list(val_top1_by_pair.values())))
            pairs_str = ", ".join(
                f"{k}={v:.1f}%" for k, v in sorted(val_top1_by_pair.items())
            )
            logger.info("Epoch 0 Val Top-1: %s | Mean: %.1f%%", pairs_str, mean_top1)
            # Always establish a valid checkpoint, including when Top-1 is 0.
            best_val_mean_top1 = mean_top1
            best_epoch = 0
            best_val_by_pair = dict(val_top1_by_pair)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": 0,
                "val_mean_top1": best_val_mean_top1,
                "val_top1_by_pair": best_val_by_pair,
                "modality_to_idx": modality_to_idx,
            }
            ckpt["loss_fn_state_dict"] = loss_fn.state_dict()
            torch.save(ckpt, output_dir / "best_model.pt")
            logger.info("Saved initial model checkpoint (epoch 0)")

    for epoch in range(epochs):
        # Enable VAE HNM after warmup
        if warmup > 0 and epoch == warmup and vae_hn_ratio > 0:
            loss_fn.hard_negative_ratio = vae_hn_ratio
            logger.info("VAE hard negative mining enabled: ratio=%s", vae_hn_ratio)

        # KL warmup annealing: linearly ramp KL weights
        if kl_warmup_epochs > 0 and (target_kl_geom > 0 or target_kl_app > 0):
            kl_progress = min(1.0, (epoch + 1) / kl_warmup_epochs)
            loss_fn.lambda_kl_geom = target_kl_geom * kl_progress
            loss_fn.lambda_kl_app = target_kl_app * kl_progress

        # GRL lambda scheduling for VAE adversarial (DANN paper: Ganin 2016)
        if config.get("adversarial") and hasattr(model, "modality_discriminator"):
            if model.modality_discriminator is not None:
                p = epoch / max(epochs - 1, 1)
                grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
                model.modality_discriminator.set_lambda(grl_lambda)
                if epoch % 10 == 0:
                    logger.info("GRL lambda: %.3f (progress=%.2f)", grl_lambda, p)

        # Train
        train_metrics = train_epoch_multi_pair(
            model, pair_datasets, loss_fn, optimizer, device,
            batch_size=batch_size, pair_sampling_alpha=pair_sampling_alpha,
            pair_local_props=pair_local_props if pair_local_props else None,
            epoch=epoch + 1,
        )
        scheduler.step()

        log_msg = (
            f"Epoch {epoch+1:3d}/{epochs}"
            f" | Loss: {train_metrics['loss']:.4f}"
            f" | Train Top-1: {train_metrics['top1']:.1f}%"
        )

        # Validate every 5 epochs and at the end of short runs.
        if ((epoch + 1) % 5 == 0 or (epoch + 1) == epochs) and val_pairs_info:
            val_top1_by_pair = {}

            for info in val_pairs_info:
                metrics = evaluate_pair(
                    model, info["cases"], extractor, pca, device,
                    info["mod_a_idx"], info["mod_b_idx"],
                    n_samples=val_n_samples, seed=val_seed,
                    pair_sampling=config.get("pair_sampling", "random"),
                    max_kpts_per_slice=config.get("max_kpts_per_slice", 500),
                    num_workers=val_extraction_workers,
                    precomputed_per_case=info.get("cached_descs"),
                )
                val_top1_by_pair[info["display"]] = metrics["top1"]

            if val_top1_by_pair:
                mean_top1 = float(np.mean(list(val_top1_by_pair.values())))
                worst_pair, worst_top1 = min(
                    val_top1_by_pair.items(), key=lambda kv: kv[1],
                )

                pairs_str = ", ".join(
                    f"{k}={v:.1f}%" for k, v in sorted(val_top1_by_pair.items())
                )
                logger.info("Val Top-1: %s", pairs_str)
                log_msg += (
                    f" | Val Mean: {mean_top1:.1f}%"
                    f" | Worst: {worst_top1:.1f}% ({worst_pair})"
                )

                # The first post-training validation replaces the epoch-0
                # fallback so short runs always emit a trained checkpoint.
                if best_epoch == 0 or mean_top1 > best_val_mean_top1:
                    best_val_mean_top1 = mean_top1
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_val_by_pair = dict(val_top1_by_pair)

                    ckpt = {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "epoch": best_epoch,
                        "val_mean_top1": best_val_mean_top1,
                        "val_top1_by_pair": best_val_by_pair,
                        "modality_to_idx": modality_to_idx,
                    }
                    ckpt["loss_fn_state_dict"] = loss_fn.state_dict()
                    torch.save(ckpt, output_dir / "best_model.pt")
                else:
                    patience_counter += 5

                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        logger.info(log_msg)

    logger.info(
        "Best Val Mean Top-1: %.1f%% at epoch %d",
        best_val_mean_top1, best_epoch,
    )
    logger.info("Best by pair: %s", best_val_by_pair)

    # ---------- Save PCA and config ----------
    with open(output_dir / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)

    final_config = config.copy()
    final_config.update({
        "best_val_mean_top1": best_val_mean_top1,
        "best_epoch": best_epoch,
        "best_val_by_pair": best_val_by_pair,
        "modality_to_idx": modality_to_idx,
        "canonical_map": {
            f"{ds}.{mod}": canon
            for (ds, mod), canon in canonical_map.items()
        } if canonical_map else {},
        "is_multi_dataset": is_multi_dataset,
        "datasets": [dict(dc) for dc in dataset_configs],
        "timestamp": datetime.now().isoformat(),
    })

    with open(output_dir / "config.json", "w") as f:
        json.dump(final_config, f, indent=2)

    logger.info("Saved to %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
