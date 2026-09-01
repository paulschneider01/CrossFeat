"""Simplified config loader for the public CrossFeat training pipeline.

It supports both legacy single-dataset configs and universal multi-dataset configs,
while pruning keys that are not used by the standard VAE training pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config.defaults import DATASET_ROOTS

_REQUIRED_KEYS = {
    "name",
    "model",
    "descriptor",
    "pca_dim",
    "pair_sampling",
    "max_kpts_per_slice",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "patience",
    "samples_per_case",
    "extraction_workers",
    "seed",
    "device",
    "output_dir",
    "exp",
}

_ALWAYS_KEEP = {
    "description",
    "datasets",
    "dataset",
    "data_root",
    "data_dir",
    "modalities",
    "pairs",
    "manifest_kwargs",
    "pair_balanced_sampling",
    "pair_sampling_alpha",
    "uniform_pair_data",
    "hidden_dim",
    "dropout",
    "d_geom",
    "d_app",
    "crosser_hidden_dim",
    "condition_decoder",
    "variational",
    "variational_app_only",
    "adversarial",
    "rootsift",
    "preprocess",
    "max_train_cases",
    "temperature",
}

_ALLOWED_DATASETS = {
    "paired_folders",
    "deliver",
    "eventscape",
    "remind",
    "resect",
    "brats",
    "whu_opt_sar",
    "qxs_saropt",
}


def _legacy_pairs_to_dicts(raw_pairs: list[Any], modalities: list[str]) -> list[dict[str, str]]:
    if raw_pairs:
        pairs: list[dict[str, str]] = []
        for p in raw_pairs:
            if isinstance(p, dict):
                src = p.get("source")
                tgt = p.get("target")
                if src and tgt:
                    pairs.append({"source": str(src), "target": str(tgt)})
            elif isinstance(p, str) and ":" in p:
                src, tgt = p.split(":", 1)
                pairs.append({"source": src.strip(), "target": tgt.strip()})
            elif isinstance(p, str) and "_" in p:
                src, tgt = p.split("_", 1)
                pairs.append({"source": src.strip(), "target": tgt.strip()})
        if pairs:
            return pairs

    if len(modalities) >= 2:
        return [{"source": modalities[0], "target": modalities[1]}]
    return []


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy and multi-dataset configs to one compact schema."""
    cfg = dict(config)

    if "datasets" not in cfg:
        dataset_name = str(cfg.get("dataset", "remind"))
        modalities = [str(m) for m in cfg.get("modalities", [])]
        pairs = _legacy_pairs_to_dicts(cfg.get("pairs", []), modalities)
        data_root = cfg.get("data_root") or cfg.get("data_dir")
        manifest_kwargs = dict(cfg.get("manifest_kwargs", {}))
        if "use_crop" in cfg and "use_crop" not in manifest_kwargs:
            manifest_kwargs["use_crop"] = bool(cfg["use_crop"])

        cfg["datasets"] = [
            {
                "name": dataset_name,
                "modalities": modalities,
                "pairs": pairs,
                "data_root": data_root,
                "manifest_kwargs": manifest_kwargs,
            }
        ]

    datasets = cfg.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Config must define a non-empty 'datasets' list.")
    for ds in datasets:
        ds_name = str(ds.get("name", ""))
        if ds_name not in _ALLOWED_DATASETS:
            allowed = ", ".join(sorted(_ALLOWED_DATASETS))
            raise ValueError(
                f"Unsupported dataset={ds_name!r}. Allowed datasets: {allowed}"
            )
        data_root = ds.get("data_root") or ds.get("data_dir")
        if not data_root:
            default_root = DATASET_ROOTS.get(ds_name, "")
            if default_root:
                ds["data_root"] = default_root
            else:
                raise ValueError(
                    f"Dataset {ds_name!r} requires a non-empty data_root in config."
                )

    normalized: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in _REQUIRED_KEYS or key in _ALWAYS_KEEP or key.startswith("vae_"):
            normalized[key] = value

    # Ensure required keys are present; keep original values when available.
    for req in _REQUIRED_KEYS:
        if req in normalized:
            continue
        if req in cfg:
            normalized[req] = cfg[req]

    missing = sorted(k for k in _REQUIRED_KEYS if k not in normalized)
    if missing:
        raise KeyError(f"Config missing required keys: {', '.join(missing)}")

    # Force stable defaults for the public pipeline.
    normalized.setdefault("model", "vae")
    normalized.setdefault("descriptor", "sift")
    normalized.setdefault("pair_sampling", "2d_sift_dual")
    normalized.setdefault("pair_balanced_sampling", False)
    normalized.setdefault("pair_sampling_alpha", 1.0)

    model_name = str(normalized.get("model", "vae")).lower()
    if model_name != "vae":
        raise ValueError(
            f"Unsupported model={model_name!r}. CrossFeat supports only model='vae'."
        )

    descriptor = str(normalized.get("descriptor", "sift")).lower()
    if descriptor != "sift":
        raise ValueError(
            f"Unsupported descriptor={descriptor!r}. CrossFeat supports only descriptor='sift'."
        )

    return normalized


def load_and_normalize_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return normalize_config(raw)


def write_config(config: dict[str, Any], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
