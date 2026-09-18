"""Two-dimensional scalar maps from uncertainty evaluation results."""

from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.image import AxesImage
from scipy.stats import rankdata

from ..evaluation import UncertaintyEvaluationResult

LOG_DISPLAY_FLOOR = 1e-14

type PanelName = Literal[
    "truth",
    "reconstruction",
    "pixel_error",
    "deformation_uncertainty",
    "parameter_laplace_uncertainty",
    "ensemble_uncertainty",
    "ensemble_mean",
    "fourier_information",
    "fourier_data_blind_variance",
    "fourier_blind_mask",
    "ensemble_fourier_variance",
    "absolute_fourier_error",
    "relative_fourier_error",
    "blind_content",
]
type ValueScale = Literal["linear", "log10"]


def plot_scalar_map_2d(
    axes: Axes,
    values: np.ndarray,
    *,
    title: str,
    color_map: str | Colormap,
    vmin: float | None = None,
    vmax: float | None = None,
    extent: tuple[float, float, float, float] | None = None,
    add_colorbar: bool = False,
) -> AxesImage:
    """
    Draw a two-dimensional scalar map on an existing axis.

    Parameters:
    -----------
    axes: Axes
        Axis that receives the image and optional colorbar.
    values: np.ndarray
        Two-dimensional scalar values.
    title: str
        Panel title.
    color_map: str | Colormap
        Matplotlib colormap name or instance.
    vmin: float | None
        Lower color limit.
    vmax: float | None
        Upper color limit.
    extent: tuple[float, float, float, float] | None
        Optional `(left, right, bottom, top)` image extent.
    add_colorbar: bool
        Whether to add a colorbar associated with this axis.

    Returns:
    --------
    AxesImage
        Matplotlib image artist added to `axes`.

    Raises:
    -------
    ValueError
        If `values` is not two-dimensional.
    """
    scalar_values = np.asarray(values)
    if scalar_values.ndim != 2:
        raise ValueError("`values` must be a two-dimensional array.")

    image = axes.imshow(
        scalar_values,
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    axes.set_title(title)
    if add_colorbar:
        axes.figure.colorbar(image, ax=axes)
    return image


def plot_result_panel(
    axes: Axes,
    evaluation: UncertaintyEvaluationResult,
    panel: PanelName,
    *,
    color_map: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    extent: tuple[float, float, float, float] | None = None,
    add_colorbar: bool = True,
    scale: ValueScale = "linear",
    percentile_rank: bool = False,
) -> AxesImage:
    """
    Draw a named evaluation panel on an existing axis.

    Parameters:
    -----------
    axes: Axes
        Axis that receives the panel.
    evaluation: UncertaintyEvaluationResult
        Evaluation containing reconstruction, UQ methods, errors, and curves.
    panel: PanelName
        Named two-dimensional result to display.
    color_map: str | Colormap | None
        Optional override for the panel's neutral default colormap.
    vmin: float | None
        Optional lower color limit.
    vmax: float | None
        Optional upper color limit.
    extent: tuple[float, float, float, float] | None
        Optional `(left, right, bottom, top)` image extent.
    add_colorbar: bool
        Whether to add a colorbar for this panel.
    scale: ValueScale
        Linear values or base-ten logarithms for display.
    percentile_rank: bool
        Convert finite values to percentile ranks from zero to one hundred.

    Returns:
    --------
    AxesImage
        Matplotlib image artist added to `axes`.
    """
    values, title, default_color_map, is_fourier = _resolve_panel(
        evaluation,
        panel,
    )
    display_values = np.fft.fftshift(values) if is_fourier else values
    if percentile_rank:
        display_values = _percentile_ranks(display_values)
        title = f"{title}\n(percentile rank)"
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 100.0
    elif scale == "log10":
        display_values = _log10_values(display_values)
        title = f"{title}\n(log10)"
    elif scale != "linear":
        raise ValueError(f"Unsupported value scale: {scale}.")

    if (
        panel == "deformation_uncertainty"
        and scale == "linear"
        and not percentile_rank
        and vmin is None
        and vmax is None
    ):
        vmin, vmax = np.percentile(display_values, (2.0, 98.0))
        title = f"{title}\n(2nd to 98th percentile limits)"
    elif panel == "blind_content" and vmin is None and vmax is None:
        absolute_limit = float(np.max(np.abs(display_values)))
        vmin, vmax = -absolute_limit, absolute_limit

    return plot_scalar_map_2d(
        axes,
        display_values,
        title=title,
        color_map=default_color_map if color_map is None else color_map,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        add_colorbar=add_colorbar,
    )


def _log10_values(values: np.ndarray) -> np.ndarray:
    """Return base-ten logarithms for non-negative display values."""
    value_array = np.asarray(values)
    if np.any(value_array < 0.0):
        raise ValueError("Logarithmic display requires non-negative values.")
    return np.log10(np.maximum(value_array, LOG_DISPLAY_FLOOR))


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return finite values as percentile ranks from zero to one hundred."""
    value_array = np.asarray(values)
    finite_mask = np.isfinite(value_array)
    finite_count = int(finite_mask.sum())
    if finite_count == 0:
        raise ValueError("Percentile ranks require at least one finite value.")

    percentiles = np.full(value_array.shape, np.nan, dtype=float)
    if finite_count == 1:
        percentiles[finite_mask] = 100.0
        return percentiles

    ranks = rankdata(value_array[finite_mask], method="average")
    percentiles[finite_mask] = 100.0 * (ranks - 1.0) / (finite_count - 1.0)
    return percentiles


def _resolve_panel(
    evaluation: UncertaintyEvaluationResult,
    panel: PanelName,
) -> tuple[np.ndarray, str, str, bool]:
    """Return panel values, title, colormap, and Fourier-domain status."""
    results = evaluation.results

    if panel == "truth":
        return evaluation.truth, "Ground truth", "gray", False
    if panel == "reconstruction":
        return results.reconstruction, "Reconstruction", "gray", False
    if panel == "pixel_error":
        return evaluation.pixel_error, "Absolute pixel error", "magma", False
    if panel == "deformation_uncertainty":
        if results.deformation is None:
            raise ValueError("Deformation uncertainty was not computed.")
        return (
            results.deformation.uncertainty_map,
            "Deformation BayesRays std",
            "plasma",
            False,
        )
    if panel == "parameter_laplace_uncertainty":
        if results.parameter_laplace is None:
            raise ValueError(
                "Parameter Laplace uncertainty was not computed."
            )
        return (
            results.parameter_laplace.pixel_standard_deviation,
            "Parameter Laplace std",
            "magma",
            False,
        )
    if panel == "ensemble_uncertainty":
        if results.ensemble is None:
            raise ValueError("Ensemble uncertainty was not computed.")
        return (
            results.ensemble.pixel_standard_deviation,
            "Ensemble std",
            "magma",
            False,
        )
    if panel == "ensemble_mean":
        if results.ensemble is None:
            raise ValueError("Ensemble uncertainty was not computed.")
        return results.ensemble.mean_image, "Ensemble mean", "gray", False
    if panel == "fourier_information":
        if results.fourier_data_blind is None:
            raise ValueError("Fourier information was not computed.")
        return (
            results.fourier_data_blind.information,
            "Fourier information",
            "viridis",
            True,
        )
    if panel == "fourier_data_blind_variance":
        if results.fourier_data_blind is None:
            raise ValueError("Fourier information was not computed.")
        return (
            results.fourier_data_blind.variance,
            "Fourier data-blind variance",
            "viridis",
            True,
        )
    if panel == "fourier_blind_mask":
        if results.fourier_data_blind is None:
            raise ValueError("Fourier information was not computed.")
        return (
            results.fourier_data_blind.blind_mask,
            "Fourier blind mask",
            "gray",
            True,
        )
    if panel == "ensemble_fourier_variance":
        if results.ensemble is None:
            raise ValueError("Ensemble uncertainty was not computed.")
        return (
            results.ensemble.fourier_variance,
            "Ensemble Fourier variance",
            "viridis",
            True,
        )
    if panel == "absolute_fourier_error":
        return (
            evaluation.absolute_fourier_error,
            "Absolute Fourier error",
            "magma",
            True,
        )
    if panel == "relative_fourier_error":
        return (
            evaluation.relative_fourier_error,
            "Relative Fourier error",
            "magma",
            True,
        )
    if panel == "blind_content":
        if results.blind_content is None:
            raise ValueError("Blind Fourier content was not computed.")
        return (
            results.blind_content,
            "Reconstruction on blind modes",
            "RdBu_r",
            False,
        )
    raise ValueError(f"Unsupported panel: {panel}.")
