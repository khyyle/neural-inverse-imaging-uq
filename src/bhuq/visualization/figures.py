"""Composed Matplotlib figures for uncertainty evaluation results."""

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from ..evaluation import UncertaintyEvaluationResult
from .maps import PanelName, ValueScale, plot_result_panel


def make_result_figure(
    evaluation: UncertaintyEvaluationResult,
    panels: Sequence[PanelName],
    *,
    columns: int | None = None,
    figure_size: tuple[float, float] | None = None,
    colorbars: bool = True,
    color_maps: dict[PanelName, str | Colormap] | None = None,
    scales: dict[PanelName, ValueScale] | None = None,
    percentile_rank_panels: set[PanelName] | None = None,
    value_limits: dict[
        PanelName,
        tuple[float | None, float | None],
    ]
    | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    """
    Compose named evaluation panels into a Matplotlib figure.

    Parameters:
    -----------
    evaluation: UncertaintyEvaluationResult
        Evaluation supplying every requested panel.
    panels: Sequence[PanelName]
        Panels in row-major display order.
    columns: int | None
        Number of subplot columns. Defaults to one row containing all panels.
    figure_size: tuple[float, float] | None
        Figure width and height in inches.
    colorbars: bool
        Whether each panel receives its own colorbar.
    color_maps: dict[PanelName, str | Colormap] | None
        Optional colormap overrides keyed by panel.
    scales: dict[PanelName, ValueScale] | None
        Optional linear or logarithmic display scales keyed by panel.
    percentile_rank_panels: set[PanelName] | None
        Panels converted to percentile ranks before display.
    value_limits: dict[PanelName, tuple[float | None, float | None]] | None
        Optional `(vmin, vmax)` overrides keyed by panel.
    extent: tuple[float, float, float, float] | None
        Optional image extent applied to every panel.

    Returns:
    --------
    figure: Figure
        Complete composed figure.
    axes: tuple[Axes, ...]
        Axes corresponding to requested panels.

    Raises:
    -------
    ValueError
        If no panels are requested or `columns` is not positive.
    """
    panel_names = tuple(panels)
    if not panel_names:
        raise ValueError("`panels` must contain at least one panel.")

    column_count = len(panel_names) if columns is None else columns
    if column_count <= 0:
        raise ValueError("`columns` must be positive.")
    row_count = math.ceil(len(panel_names) / column_count)
    resolved_figure_size = figure_size or (
        3.8 * column_count,
        3.4 * row_count,
    )

    figure, axes_grid = plt.subplots(
        row_count,
        column_count,
        figsize=resolved_figure_size,
        squeeze=False,
        constrained_layout=True,
    )
    flattened_axes = tuple(axes_grid.reshape(-1))
    selected_axes = flattened_axes[: len(panel_names)]
    color_map_overrides = color_maps or {}
    scale_overrides = scales or {}
    percentile_panels = percentile_rank_panels or set()
    value_limit_overrides = value_limits or {}
    for axes, panel in zip(selected_axes, panel_names, strict=True):
        vmin, vmax = value_limit_overrides.get(panel, (None, None))
        plot_result_panel(
            axes,
            evaluation,
            panel,
            color_map=color_map_overrides.get(panel),
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            add_colorbar=colorbars,
            scale=scale_overrides.get(panel, "linear"),
            percentile_rank=panel in percentile_panels,
        )
    for unused_axes in flattened_axes[len(panel_names) :]:
        unused_axes.set_visible(False)
    return figure, selected_axes


def make_uncertainty_figure(
    evaluation: UncertaintyEvaluationResult,
    *,
    columns: int | None = None,
    colorbars: bool = True,
    color_maps: dict[PanelName, str | Colormap] | None = None,
    scale: ValueScale = "log10",
    percentile_rank: bool = False,
    percentile_color_map: str | Colormap = "viridis",
    extent: tuple[float, float, float, float] | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    """
    Compose every available image-space uncertainty method.

    Raw uncertainty maps use logarithmic display by default. Set
    `percentile_rank=True` to compare only their spatial rankings with a
    shared zero-to-one-hundred colorbar.
    """
    panels: list[PanelName] = []
    if evaluation.results.deformation is not None:
        panels.append("deformation_uncertainty")
    if evaluation.results.parameter_laplace is not None:
        panels.append("parameter_laplace_uncertainty")
    if evaluation.results.ensemble is not None:
        panels.append("ensemble_uncertainty")
    panel_scales = {panel: scale for panel in panels}
    panel_color_maps = color_maps
    percentile_panels = None
    panel_colorbars = colorbars
    if percentile_rank:
        panel_scales = {}
        panel_color_maps = {
            panel: percentile_color_map
            for panel in panels
        }
        percentile_panels = set(panels)
        panel_colorbars = False

    figure, axes = make_result_figure(
        evaluation,
        panels,
        columns=columns,
        colorbars=panel_colorbars,
        color_maps=panel_color_maps,
        scales=panel_scales,
        percentile_rank_panels=percentile_panels,
        extent=extent,
    )
    if percentile_rank and colorbars:
        figure.colorbar(
            axes[0].images[0],
            ax=list(axes),
            label="Percentile rank",
        )
    return figure, axes


def make_fourier_figure(
    evaluation: UncertaintyEvaluationResult,
    *,
    columns: int | None = None,
    include_errors: bool = True,
    colorbars: bool = True,
    color_maps: dict[PanelName, str | Colormap] | None = None,
    scale: ValueScale = "log10",
) -> tuple[Figure, tuple[Axes, ...]]:
    """Compose Fourier diagnostics with logarithmic continuous maps."""
    panels: list[PanelName] = []
    if evaluation.results.fourier_data_blind is not None:
        panels.extend(
            (
                "fourier_information",
                "fourier_data_blind_variance",
                "fourier_blind_mask",
            )
        )
    if evaluation.results.ensemble is not None:
        panels.append("ensemble_fourier_variance")
    if include_errors:
        panels.extend(
            (
                "absolute_fourier_error",
                "relative_fourier_error",
            )
        )
    panel_scales = {
        panel: scale
        for panel in panels
        if panel != "fourier_blind_mask"
    }
    return make_result_figure(
        evaluation,
        panels,
        columns=columns,
        colorbars=colorbars,
        color_maps=color_maps,
        scales=panel_scales,
    )
