from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from evaluate import _json_safe, _resolve_modality_indices, parse_args
from train_universal import _prepare_output_dir


def test_evaluation_defaults_to_all_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--model_dir", "model"])

    assert parse_args().max_cases == 0


def test_unknown_modalities_fail_instead_of_using_first_pair() -> None:
    config = {"modality_to_idx": {"paired_folders.a": 0, "paired_folders.b": 1}}

    with pytest.raises(ValueError, match="Available modality keys"):
        _resolve_modality_indices(config, "paired_folders", ["mistyped", "b"])


def test_json_output_replaces_non_finite_values() -> None:
    safe = _json_safe({"finite": 1.0, "infinite": float("inf"), "nan": float("nan")})

    assert safe == {"finite": 1.0, "infinite": None, "nan": None}
    assert json.dumps(safe, allow_nan=False)


def test_training_refuses_non_empty_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "experiments" / "crossfeat" / "existing"
    output_dir.mkdir(parents=True)
    (output_dir / "best_model.pt").write_bytes(b"existing")
    config = {
        "output_dir": str(tmp_path / "experiments"),
        "exp": "crossfeat",
        "name": "existing",
    }

    with pytest.raises(FileExistsError, match="Choose a new config 'name'"):
        _prepare_output_dir(config)


def test_training_can_retry_diagnostics_only_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "experiments" / "crossfeat" / "retry"
    (output_dir / "logs").mkdir(parents=True)
    (output_dir / "logs" / "failed.log").write_text("failed", encoding="utf-8")
    (output_dir / "config_source.yaml").write_text("name: retry\n", encoding="utf-8")
    config = {
        "output_dir": str(tmp_path / "experiments"),
        "exp": "crossfeat",
        "name": "retry",
    }

    assert _prepare_output_dir(config) == output_dir
