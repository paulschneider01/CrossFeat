"""
Visualization module for CrossFeat.

Provides functions for creating visual sanity checks of cross-modal matching.
"""

from src.viz.evaluation import (
    VisualizationData,
    plot_matches_multiview,
    plot_tre_distribution,
    plot_crossing_comparison,
    plot_tta_comparison,
    create_summary_figure,
)
from src.viz.utils import (
    COLORS,
    normalize_for_display,
    get_slice,
    get_slice_coords,
    format_metrics_text,
)

__all__ = [
    # Data classes
    "VisualizationData",
    # Plotting functions
    "plot_matches_multiview",
    "plot_tre_distribution",
    "plot_crossing_comparison",
    "plot_tta_comparison",
    "create_summary_figure",
    # Utilities
    "COLORS",
    "normalize_for_display",
    "get_slice",
    "get_slice_coords",
    "format_metrics_text",
]
