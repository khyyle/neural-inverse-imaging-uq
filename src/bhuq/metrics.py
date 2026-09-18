"""Metrics for evaluating uncertainty maps against known reconstruction error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class SparsificationResult:
    """
    Store a sparsification curve and its area above the oracle curve.

    Parameters:
    -----------
    removed_fractions: np.ndarray
        Fractions of values removed from the error calculation.
    uncertainty_ranked_error: np.ndarray
        Normalized remaining error after removing values from highest to
        lowest predicted uncertainty.
    oracle_ranked_error: np.ndarray
        Normalized remaining error after removing values from highest to
        lowest true error.
    area_under_sparsification_error: float
        Trapezoidal area between the uncertainty-ranked and oracle curves.
        Lower values indicate a better uncertainty ranking.
    """

    removed_fractions: np.ndarray
    uncertainty_ranked_error: np.ndarray
    oracle_ranked_error: np.ndarray
    area_under_sparsification_error: float

    @property
    def sparsification_error(self) -> np.ndarray:
        """Return the pointwise gap above the oracle sparsification curve."""
        return self.uncertainty_ranked_error - self.oracle_ranked_error


def sparsification_curve(
    uncertainty: np.ndarray,
    error: np.ndarray,
    *,
    number_of_steps: int = 60,
    maximum_removed_fraction: float = 0.98,
) -> SparsificationResult:
    """
    Measure how well uncertainty ranks locations with high true error.

    At each removal fraction, values with the greatest uncertainty are
    discarded and the mean error of the remaining values is recorded. The
    oracle curve performs the same procedure using true error for ranking.

    Parameters:
    -----------
    uncertainty: np.ndarray
        Predicted uncertainty values. Larger values are removed first.
    error: np.ndarray
        Non-negative reference errors with the same number of elements as
        `uncertainty`.
    number_of_steps: int
        Number of removal fractions sampled along the curve.
    maximum_removed_fraction: float
        Largest fraction removed. Must be at least zero and less than one.

    Returns:
    --------
    SparsificationResult
        Normalized uncertainty and oracle curves together with their area
        difference.

    Raises:
    -------
    ValueError
        If inputs are empty, have different sizes, contain non-finite values,
        or use invalid curve settings.
    """
    uncertainty_values = np.asarray(uncertainty).reshape(-1)
    error_values = np.asarray(error).reshape(-1)

    if uncertainty_values.size == 0:
        raise ValueError("`uncertainty` and `error` must not be empty.")
    if uncertainty_values.size != error_values.size:
        raise ValueError("`uncertainty` and `error` must have the same size.")
    if not np.all(np.isfinite(uncertainty_values)):
        raise ValueError("`uncertainty` must contain finite values.")
    if not np.all(np.isfinite(error_values)):
        raise ValueError("`error` must contain finite values.")
    if np.any(error_values < 0.0):
        raise ValueError("`error` must be non-negative.")
    if number_of_steps < 2:
        raise ValueError("`number_of_steps` must be at least two.")
    if not 0.0 <= maximum_removed_fraction < 1.0:
        raise ValueError("`maximum_removed_fraction` must be in [0, 1).")

    initial_mean_error = float(error_values.mean())
    if initial_mean_error == 0.0:
        raise ValueError("A sparsification curve is undefined when all errors are zero.")

    removed_fractions = np.linspace(
        0.0,
        maximum_removed_fraction,
        number_of_steps,
    )
    uncertainty_order = np.argsort(-uncertainty_values)
    oracle_order = np.argsort(-error_values)

    def remaining_mean_error(order: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                error_values[
                    order[int(fraction * error_values.size) :]
                ].mean()
                for fraction in removed_fractions
            ]
        )

    uncertainty_ranked_error = remaining_mean_error(uncertainty_order)
    oracle_ranked_error = remaining_mean_error(oracle_order)
    uncertainty_ranked_error /= initial_mean_error
    oracle_ranked_error /= initial_mean_error
    area = np.trapezoid(
        uncertainty_ranked_error - oracle_ranked_error,
        removed_fractions,
    )
    return SparsificationResult(
        removed_fractions=removed_fractions,
        uncertainty_ranked_error=uncertainty_ranked_error,
        oracle_ranked_error=oracle_ranked_error,
        area_under_sparsification_error=float(area),
    )


def area_under_sparsification_error(
    uncertainty: np.ndarray,
    error: np.ndarray,
    *,
    number_of_steps: int = 60,
    maximum_removed_fraction: float = 0.98,
) -> float:
    """
    Return only the area between uncertainty and oracle sparsification curves.

    Parameters:
    -----------
    uncertainty: np.ndarray
        Predicted uncertainty values.
    error: np.ndarray
        Reference errors with the same number of elements.
    number_of_steps: int
        Number of removal fractions sampled along the curve.
    maximum_removed_fraction: float
        Largest fraction removed.

    Returns:
    --------
    float
        Area under the sparsification error curve. Zero is a perfect ranking.
    """
    return sparsification_curve(
        uncertainty,
        error,
        number_of_steps=number_of_steps,
        maximum_removed_fraction=maximum_removed_fraction,
    ).area_under_sparsification_error


def spearman_rank_correlation(
    uncertainty: np.ndarray,
    error: np.ndarray,
) -> float:
    """
    Compute rank correlation between predicted uncertainty and true error.

    Parameters:
    -----------
    uncertainty: np.ndarray
        Predicted uncertainty values.
    error: np.ndarray
        Reference errors with the same number of elements.

    Returns:
    --------
    float
        Spearman correlation. One is a perfect increasing rank relationship;
        minus one is a perfect decreasing relationship.

    Raises:
    -------
    ValueError
        If the arrays have different sizes.
    """
    uncertainty_values = np.asarray(uncertainty).reshape(-1)
    error_values = np.asarray(error).reshape(-1)
    if uncertainty_values.size != error_values.size:
        raise ValueError("`uncertainty` and `error` must have the same size.")
    correlation = spearmanr(uncertainty_values, error_values).correlation
    return float(correlation)


def fourier_error_maps(
    reconstruction: np.ndarray,
    truth: np.ndarray,
    *,
    relative_floor_fraction: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute absolute and relative reconstruction errors per Fourier mode.

    Parameters:
    -----------
    reconstruction: np.ndarray
        Reconstructed image.
    truth: np.ndarray
        Ground-truth image with the same shape.
    relative_floor_fraction: float
        Fraction of the maximum true Fourier magnitude added to the relative
        error denominator.

    Returns:
    --------
    absolute_error: np.ndarray
        Absolute difference between reconstructed and true Fourier
        coefficients.
    relative_error: np.ndarray
        Absolute error divided by the floored true Fourier magnitude.
    """
    reconstruction_array = np.asarray(reconstruction)
    truth_array = np.asarray(truth)
    if reconstruction_array.shape != truth_array.shape:
        raise ValueError("`reconstruction` and `truth` must have the same shape.")
    if relative_floor_fraction < 0.0:
        raise ValueError("`relative_floor_fraction` must be non-negative.")

    reconstruction_fourier = np.fft.fft2(reconstruction_array)
    truth_fourier = np.fft.fft2(truth_array)
    absolute_error = np.abs(reconstruction_fourier - truth_fourier)
    denominator_floor = (
        relative_floor_fraction * np.abs(truth_fourier).max()
    )
    relative_error = absolute_error / (
        np.abs(truth_fourier) + denominator_floor
    )
    return absolute_error, relative_error


def masked_fourier_power_fraction(
    image: np.ndarray,
    fourier_mask: np.ndarray,
) -> float:
    """
    Measure the fraction of image Fourier power selected by a mask.

    Parameters:
    -----------
    image: np.ndarray
        Two-dimensional image.
    fourier_mask: np.ndarray
        Boolean mask with the same shape in unshifted FFT ordering.

    Returns:
    --------
    float
        Selected Fourier power divided by total Fourier power.
    """
    image_array = np.asarray(image)
    mask_array = np.asarray(fourier_mask, dtype=bool)
    if image_array.shape != mask_array.shape:
        raise ValueError("`image` and `fourier_mask` must have the same shape.")

    fourier_power = np.abs(np.fft.fft2(image_array)) ** 2
    total_power = float(fourier_power.sum())
    if total_power == 0.0:
        return 0.0
    return float(fourier_power[mask_array].sum() / total_power)


def masked_fourier_content(
    image: np.ndarray,
    fourier_mask: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the image content selected by a Fourier-domain mask.

    Parameters:
    -----------
    image: np.ndarray
        Two-dimensional image.
    fourier_mask: np.ndarray
        Boolean mask with the same shape in unshifted FFT ordering.

    Returns:
    --------
    np.ndarray
        Real inverse FFT after zeroing coefficients outside the mask.
    """
    image_array = np.asarray(image)
    mask_array = np.asarray(fourier_mask, dtype=bool)
    if image_array.shape != mask_array.shape:
        raise ValueError("`image` and `fourier_mask` must have the same shape.")
    selected_coefficients = np.where(
        mask_array,
        np.fft.fft2(image_array),
        0.0,
    )
    return np.real(np.fft.ifft2(selected_coefficients))


def masked_fourier_content_correlation(
    reconstruction: np.ndarray,
    truth: np.ndarray,
    fourier_mask: np.ndarray,
) -> float:
    """
    Compare reconstructed and true content selected by a Fourier mask.

    Parameters:
    -----------
    reconstruction: np.ndarray
        Reconstructed image.
    truth: np.ndarray
        Ground-truth image with the same shape.
    fourier_mask: np.ndarray
        Boolean Fourier-domain mask with the same shape.

    Returns:
    --------
    float
        Pearson correlation between the two masked image-space components.
    """
    reconstructed_content = masked_fourier_content(
        reconstruction,
        fourier_mask,
    )
    true_content = masked_fourier_content(truth, fourier_mask)
    return float(
        np.corrcoef(
            reconstructed_content.reshape(-1),
            true_content.reshape(-1),
        )[0, 1]
    )
