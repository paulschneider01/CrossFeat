from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _textured_pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(160, 160), dtype=np.uint8)
    for i in range(8):
        center = (20 + (i * 17) % 120, 20 + (i * 29) % 120)
        cv2.circle(image, center, 5 + i % 4, int(25 + i * 25), 2)
    paired = cv2.convertScaleAbs(cv2.GaussianBlur(image, (3, 3), 0), alpha=0.9, beta=15)
    return image, paired


def _write_dataset(root: Path) -> None:
    counts = {"train": 3, "val": 1, "test": 1}
    seed = 0
    for split, count in counts.items():
        for index in range(count):
            image_a, image_b = _textured_pair(seed)
            seed += 1
            relative = Path("scene") / f"sample_{index:02d}.png"
            for modality, image in (("a", image_a), ("b", image_b)):
                path = root / split / modality / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(path), image)


def _run(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_train_then_evaluate_paired_folders(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_dataset(data_root)

    output_root = tmp_path / "experiments"
    config = {
        "name": "smoke",
        "datasets": [{
            "name": "paired_folders",
            "data_root": str(data_root),
            "modalities": ["a", "b"],
            "pairs": [{"source": "a", "target": "b"}],
        }],
        "model": "vae",
        "d_geom": 8,
        "d_app": 8,
        "hidden_dim": 32,
        "dropout": 0.0,
        "condition_decoder": True,
        "variational": True,
        "variational_app_only": True,
        "adversarial": False,
        "descriptor": "sift",
        "pca_dim": 8,
        "pair_sampling": "2d_sift_dual",
        "rootsift": True,
        "max_kpts_per_slice": 128,
        "epochs": 1,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "patience": 5,
        "samples_per_case": 64,
        "extraction_workers": 1,
        "seed": 42,
        "device": "cpu",
        "output_dir": str(output_root),
        "exp": "crossfeat",
        "vae_recon": 1.0,
        "vae_geom": 0.5,
        "vae_crossing": 1.0,
        "vae_contrastive": 0.1,
        "vae_kl_geom": 0.0,
        "vae_kl_app": 0.01,
        "vae_adversarial": 0.0,
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    _run(
        "train.py",
        "--config", str(config_path),
        "--normalized_config_out", str(tmp_path / "normalized.yaml"),
        env=env,
    )

    model_dir = output_root / "crossfeat" / "smoke"
    assert (model_dir / "best_model.pt").is_file()
    assert (model_dir / "pca.pkl").is_file()
    assert (model_dir / "config.json").is_file()
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1

    output_json = tmp_path / "evaluation.json"
    _run(
        "evaluate.py",
        "--model_dir", str(model_dir),
        "--dataset", "paired_folders",
        "--data_root", str(data_root),
        "--modalities", "a", "b",
        "--split", "test",
        "--n_samples", "64",
        "--max_kpts_per_slice", "128",
        "--device", "cpu",
        "--output_json", str(output_json),
        env=env,
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["aggregate"]["n_cases"] == 1
    assert result["aggregate"]["n_success"] == 1
    assert result["cases"][0]["error"] == ""
