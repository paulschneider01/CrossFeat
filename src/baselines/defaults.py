from __future__ import annotations

import argparse


def add_baseline_args(parser: argparse.ArgumentParser) -> None:
    """Register minimal baseline-related CLI arguments.

    The upload package only needs the no-cross baseline.
    """

    group = parser.add_argument_group("Baselines")
    group.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=["none", "nocross"],
        help="Also run a baseline method for comparison (default: none). "
        "'nocross' matches raw descriptors without crossing.",
    )

    # Visualization.
    group.add_argument(
        "--viz_baseline",
        action="store_true",
        help="Also visualize baseline results (requires --baseline).",
    )
