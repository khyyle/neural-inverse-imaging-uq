"""Uncertainty estimates from independently trained image reconstructions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnsembleUncertainty:
    """
    Parameters:
    -----------
    mean_image: np.ndarray
        Mean reconstruction with shape `(height, width)`.
    pixel_standard_deviation: np.ndarray
        Standard deviation across ensemble members at each pixel.
    fourier_variance: np.ndarray
        Variance across ensemble members at each Fourier mode.
    """

    mean_image: np.ndarray
    pixel_standard_deviation: np.ndarray
    fourier_variance: np.ndarray


def ensemble_uncertainty(member_images: np.ndarray) -> EnsembleUncertainty:
    """
    Compute empirical uncertainty from independently fitted images.

    Parameters:
    -----------
    member_images: np.ndarray
        Reconstructions with shape `(number_of_members, height, width)`.

    Returns:
    --------
    EnsembleUncertainty
        Ensemble mean, per-pixel standard deviation, and per-mode Fourier
        variance.

    Raises:
    -------
    ValueError
        If fewer than two two-dimensional member images are provided.
    """
    images = np.asarray(member_images)
    if images.ndim != 3:
        raise ValueError(
            "`member_images` must have shape "
            "(number_of_members, height, width)."
        )
    if images.shape[0] < 2:
        raise ValueError("At least two ensemble members are required.")

    mean_image = images.mean(axis=0)
    centered_images = images - mean_image

    # For centered member d_s, Fourier covariance is F C F^H. Mode k has
    # variance mean_s[(F d_s)_k conj((F d_s)_k)] = mean_s[|(F d_s)_k|^2],
    # so its diagonal can be computed without constructing image covariance C.
    fourier_deviations = np.fft.fft2(
        centered_images,
        axes=(-2, -1),
    )
    return EnsembleUncertainty(
        mean_image=mean_image,
        pixel_standard_deviation=images.std(axis=0),
        fourier_variance=np.mean(
            np.abs(fourier_deviations) ** 2,
            axis=0,
        ),
    )
