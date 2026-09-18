"""Sparsification curves from uncertainty evaluation results."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..evaluation import UncertaintyEvaluationResult

type SparsificationTarget = Literal[
    "image",
    "fourier_absolute",
    "fourier_relative",
]


def plot_sparsification(
    axes: Axes,
    evaluation: UncertaintyEvaluationResult,
    *,
    target: SparsificationTarget = "image",
) -> tuple[Line2D, ...]:
    """
    Draw uncertainty-ranked and oracle sparsification curves.

    Parameters:
    -----------
    axes: Axes
        Axis that receives the curves and legend.
    evaluation: UncertaintyEvaluationResult
        Evaluation containing full sparsification results.
    target: SparsificationTarget
        Error domain represented by the curves.

    Returns:
    --------
    tuple[Line2D, ...]
        Method curves followed by the shared oracle curve.

    Raises:
    -------
    ValueError
        If no curves exist for the target or retained curve samples are
        unavailable.
    """
    target_prefix = f"{target}/"
    selected_results = [
        (name.removeprefix(target_prefix), result)
        for name, result in evaluation.sparsification_results.items()
        if name.startswith(target_prefix)
    ]
    if not selected_results:
        raise ValueError(
            f"No sparsification results are available for `{target}`."
        )

    lines = []
    for method_name, result in selected_results:
        curve_values = (
            result.removed_fractions,
            result.uncertainty_ranked_error,
            result.oracle_ranked_error,
        )
        if not all(np.all(np.isfinite(values)) for values in curve_values):
            raise ValueError(
                f"Sparsification curve samples are unavailable for "
                f"`{target}/{method_name}`."
            )
        label = (
            f"{_format_method_name(method_name)} "
            f"(AUSE {result.area_under_sparsification_error:.3f})"
        )
        line, = axes.plot(
            result.removed_fractions,
            result.uncertainty_ranked_error,
            label=label,
        )
        lines.append(line)

    first_result = selected_results[0][1]
    oracle_line, = axes.plot(
        first_result.removed_fractions,
        first_result.oracle_ranked_error,
        color="black",
        linestyle="--",
        label="Oracle",
    )
    lines.append(oracle_line)
    axes.set_title(_sparsification_title(target))
    axes.set_xlabel("Fraction removed")
    axes.set_ylabel("Normalized remaining error")
    axes.legend()
    return tuple(lines)


def make_sparsification_figure(
    evaluation: UncertaintyEvaluationResult,
    *,
    target: SparsificationTarget = "image",
    figure_size: tuple[float, float] = (6.0, 4.0),
) -> tuple[Figure, Axes]:
    """Create a complete sparsification comparison figure."""
    figure, axes = plt.subplots(
        figsize=figure_size,
        constrained_layout=True,
    )
    plot_sparsification(axes, evaluation, target=target)
    return figure, axes


def _format_method_name(method_name: str) -> str:
    """Return a readable method label."""
    labels = {
        "deformation": "Deformation BayesRays",
        "parameter_laplace": "Parameter Laplace",
        "ensemble": "Ensemble",
        "fourier_data_blind": "Fourier data blind",
    }
    return labels.get(method_name, method_name.replace("_", " ").title())


def _sparsification_title(target: SparsificationTarget) -> str:
    """Return the title for one sparsification target."""
    titles = {
        "image": "Image-space sparsification",
        "fourier_absolute": "Absolute Fourier-error sparsification",
        "fourier_relative": "Relative Fourier-error sparsification",
    }
    return titles[target]
