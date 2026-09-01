#!/usr/bin/env python3
"""Minimal evaluation entrypoint for the uploadable CrossFeat pipeline."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from config.defaults import DATASET_ROOTS
from evaluate_core import evaluate_case
from src.utils.model_loading import load_pretrained_model
from src.descriptor import SIFTDescriptor
from src.io.case_adapter import load_cases_from_manifest
from src.io.dataset_registry import build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CrossFeat checkpoint.")
    parser.add_argument("--model_dir", type=Path, required=True, help="Directory with best_model.pt/pca.pkl/config.json")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name override.")
    parser.add_argument("--data_root", type=Path, default=None, help="Dataset root override.")
    parser.add_argument("--modalities", nargs=2, default=None, help="Evaluate this modality pair (mod_a mod_b).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_cases", type=int, default=0, help="Limit evaluated cases (0 = all).")
    parser.add_argument("--case_ids", nargs="*", default=None, help="Optional explicit case IDs to evaluate.")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--max_kpts_per_slice", type=int, default=None)
    parser.add_argument("--ratio_thresh", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output_json", type=Path, default=Path("results/eval_summary.json"))
    return parser.parse_args()


def _resolve_eval_setup(config: dict[str, Any], args: argparse.Namespace) -> tuple[str, Path, list[str], dict[str, Any]]:
    datasets = config.get("datasets", [])

    if args.dataset:
        dataset = args.dataset
    elif datasets:
        dataset = str(datasets[0]["name"])
    else:
        dataset = str(config.get("dataset", "resect"))

    ds_entry = None
    for ds in datasets:
        if str(ds.get("name")) == dataset:
            ds_entry = ds
            break

    if args.modalities:
        modalities = [str(args.modalities[0]), str(args.modalities[1])]
    elif ds_entry and ds_entry.get("pairs"):
        first_pair = ds_entry["pairs"][0]
        modalities = [str(first_pair["source"]), str(first_pair["target"])]
    elif config.get("modalities") and len(config["modalities"]) >= 2:
        modalities = [str(config["modalities"][0]), str(config["modalities"][1])]
    else:
        raise ValueError("Could not infer modalities; pass --modalities")

    if args.data_root is not None:
        data_root = args.data_root
    elif ds_entry and ds_entry.get("data_root"):
        data_root = Path(str(ds_entry["data_root"]))
    elif config.get("data_root"):
        data_root = Path(str(config["data_root"]))
    else:
        default_root = DATASET_ROOTS.get(dataset, "")
        if not default_root:
            raise ValueError(f"No data_root for dataset={dataset!r}; pass --data_root")
        data_root = Path(default_root)

    manifest_kwargs = dict(ds_entry.get("manifest_kwargs", {})) if ds_entry else {}
    return dataset, data_root, modalities, manifest_kwargs


def _resolve_modality_indices(config: dict[str, Any], dataset: str, modalities: list[str]) -> tuple[int, int]:
    mapping = config.get("modality_to_idx", {})
    if mapping:
        k0 = f"{dataset}.{modalities[0]}"
        k1 = f"{dataset}.{modalities[1]}"
        if k0 in mapping and k1 in mapping:
            return int(mapping[k0]), int(mapping[k1])
        if modalities[0] in mapping and modalities[1] in mapping:
            return int(mapping[modalities[0]]), int(mapping[modalities[1]])
        available = ", ".join(sorted(str(key) for key in mapping))
        raise ValueError(
            f"Modalities {modalities!r} are not present in the trained model. "
            f"Available modality keys: {available}"
        )

    if "modalities" in config:
        mods = [str(m) for m in config["modalities"]]
        try:
            return mods.index(modalities[0]), mods.index(modalities[1])
        except ValueError as exc:
            raise ValueError(
                f"Modalities {modalities!r} are not present in the trained model. "
                f"Available modalities: {mods}"
            ) from exc

    raise ValueError(
        "The model config does not contain modality indices; "
        "retrain it with the public CrossFeat pipeline."
    )


def _extractor_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    descriptor = str(config.get("descriptor", "sift"))
    kwargs: dict[str, Any] = {}
    if descriptor == "sift":
        kwargs["rootsift"] = bool(config.get("rootsift", False))
        kwargs["compute_orientation"] = not bool(config.get("no_compute_orientation", False))
        preprocess = str(config.get("preprocess", "none"))
        if preprocess != "none":
            kwargs["preprocess"] = preprocess
    return kwargs


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if not r.get("error")]
    if not ok:
        return {"n_cases": len(results), "n_success": 0}

    def avg(key: str) -> float:
        return float(mean(float(r[key]) for r in ok))

    return {
        "n_cases": len(results),
        "n_success": len(ok),
        "top1_acc": avg("top1_acc"),
        "top5_acc": avg("top5_acc"),
        "top10_acc": avg("top10_acc"),
        "inlier_ratio": avg("inlier_ratio"),
        "tre_mean": avg("tre_mean"),
        "tre_median": avg("tre_median"),
        "num_matches": avg("num_matches"),
        "num_inliers": avg("num_inliers"),
        "time_s": avg("time_s"),
    }


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so output remains strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()

    model, pca, config = load_pretrained_model(args.model_dir, device=args.device, verbose=False)
    dataset, data_root, modalities, manifest_kwargs = _resolve_eval_setup(config, args)
    mod_a_idx, mod_b_idx = _resolve_modality_indices(config, dataset, modalities)

    manifest = build_manifest(dataset, data_root, modalities, **manifest_kwargs)
    cases = load_cases_from_manifest(manifest, args.split)

    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [c for c in cases if c.case_id in wanted]

    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    if not cases:
        raise RuntimeError("No cases found for evaluation")

    descriptor = str(config.get("descriptor", "sift")).lower()
    if descriptor != "sift":
        raise ValueError(
            f"Unsupported descriptor={descriptor!r}. CrossFeat supports only descriptor='sift'."
        )
    extractor = SIFTDescriptor(**_extractor_kwargs_from_config(config))

    pair_sampling = str(config.get("pair_sampling", "2d_sift_dual"))
    max_kpts = int(config.get("max_kpts_per_slice", 500)) if args.max_kpts_per_slice is None else int(args.max_kpts_per_slice)

    rows: list[dict[str, Any]] = []
    for case in cases:
        res = evaluate_case(
            case=case,
            modalities=modalities,
            extractor=extractor,
            pca=pca,
            model=model,
            device=args.device,
            rotate_deg=0.0,
            translate_mm=0.0,
            affine_scale=0.0,
            affine_shear_deg=0.0,
            deterministic_transform=False,
            n_samples=int(args.n_samples),
            inlier_threshold=5.0,
            seed=int(args.seed),
            adapter=None,
            return_viz_data=False,
            two_d_only=True,
            pair_sampling=pair_sampling,
            max_kpts_per_slice=max_kpts,
            mod_a_idx=mod_a_idx,
            mod_b_idx=mod_b_idx,
            ratio_thresh=float(args.ratio_thresh),
        )
        rows.append(asdict(res))

    summary = {
        "model_dir": str(args.model_dir),
        "dataset": dataset,
        "data_root": str(data_root),
        "modalities": modalities,
        "split": args.split,
        "aggregate": _aggregate(rows),
        "cases": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    agg = summary["aggregate"]
    print(f"Evaluated {agg['n_success']}/{agg['n_cases']} cases")
    if agg.get("n_success", 0) > 0:
        print(
            f"Top-1={agg['top1_acc']*100:.2f}% | "
            f"Inlier={agg['inlier_ratio']*100:.2f}% | "
            f"TRE={agg['tre_median']:.3f}"
        )
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
