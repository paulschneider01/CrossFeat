"""Model loading utilities for the upload pipeline."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from sklearn.decomposition import PCA

from src.cross.models import DisentangledVAECrosser


def _remap_legacy_vae_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    remapped: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        if "app_crosser.film_net.2." in key:
            new_key = key.replace("app_crosser.film_net.2.", "app_crosser.film_net.3.")
        elif "app_crosser.transform.2." in key:
            new_key = key.replace("app_crosser.transform.2.", "app_crosser.transform.3.")
        remapped[new_key] = value
    return remapped


def _resolve_vae_architecture_flags(
    config: Dict[str, Any],
    checkpoint: Dict[str, Any],
    state_dict: Dict[str, Any],
) -> Tuple[int, bool, bool]:
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    fsq_levels = int(config.get("fsq_levels", checkpoint_config.get("fsq_levels", 0)) or 0)
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


def load_config_only(model_dir: Path) -> Dict[str, Any]:
    with open(Path(model_dir) / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_pretrained_model(
    model_dir: Path,
    model_name: str = "best_model",
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[Any, PCA, Dict[str, Any]]:
    model_dir = Path(model_dir)

    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(model_dir / "pca.pkl", "rb") as f:
        pca = pickle.load(f)

    model_type = str(config.get("model", "vae")).lower()
    if model_type != "vae":
        raise ValueError(
            f"Unsupported model={model_type!r}. CrossFeat supports only model='vae'."
        )

    if "modalities" in config:
        modalities = config["modalities"]
    elif "datasets" in config:
        seen = set()
        modalities = []
        for ds in config["datasets"]:
            for mod in ds.get("modalities", []):
                if mod not in seen:
                    seen.add(mod)
                    modalities.append(mod)
    else:
        raise KeyError("Config has neither 'modalities' nor 'datasets'")

    checkpoint = torch.load(
        model_dir / f"{model_name}.pt",
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", {})
    fsq_levels, variational, variational_app_only = _resolve_vae_architecture_flags(
        config=config,
        checkpoint=checkpoint,
        state_dict=state_dict,
    )

    modality_to_idx = config.get("modality_to_idx")
    num_modalities = len(modality_to_idx) if modality_to_idx else len(modalities)
    model = DisentangledVAECrosser(
        descriptor_dim=int(config["pca_dim"]),
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
        fsq_levels=fsq_levels,
        variational=variational,
        variational_app_only=variational_app_only,
        use_grad_head=config.get("vae_grad_head", 0.0) > 0,
        use_modality_heads=config.get("vae_modality_heads", 0.0) > 0,
        use_coord_head=config.get("vae_coord_head", 0.0) > 0,
        crosser_hidden_dim=config.get("crosser_hidden_dim", None) or None,
        disentangle=config.get("disentangle", True),
    )

    state_dict = _remap_legacy_vae_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded VAE model ({n_params:,} params)")
        val_top1 = checkpoint.get("val_top1")
        if val_top1 is not None:
            print(f"Best val Top-1 during training: {val_top1:.2f}%")

    return model, pca, config
