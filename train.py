#!/usr/bin/env python3
"""Minimal training entrypoint for the uploadable CrossFeat pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config.simple_config import load_and_normalize_config, write_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CrossFeat with a simplified config frontend.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    parser.add_argument(
        "--normalized_config_out",
        type=Path,
        default=None,
        help="Optional normalized-config path (default: .tmp/normalized_<name>.yaml).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only write and validate normalized config; do not launch training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_normalize_config(args.config)

    if args.normalized_config_out is not None:
        normalized_path = args.normalized_config_out
    else:
        normalized_path = Path(".tmp") / f"normalized_{args.config.stem}.yaml"

    write_config(config, normalized_path)
    print(f"Normalized config written to: {normalized_path}")

    if args.dry_run:
        return

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_universal.py"),
        "--config",
        str(normalized_path),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
