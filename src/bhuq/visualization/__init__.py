"""Composable figures for uncertainty evaluation results."""

from .figures import (
    make_fourier_figure,
    make_result_figure,
    make_uncertainty_figure,
)
from .maps import (
    PanelName,
    ValueScale,
    plot_result_panel,
    plot_scalar_map_2d,
)
from .sparsification import (
    SparsificationTarget,
    make_sparsification_figure,
    plot_sparsification,
)

__all__ = [
    "PanelName",
    "SparsificationTarget",
    "ValueScale",
    "make_fourier_figure",
    "make_result_figure",
    "make_sparsification_figure",
    "make_uncertainty_figure",
    "plot_result_panel",
    "plot_scalar_map_2d",
    "plot_sparsification",
]