"""
Measure Fourier-mode identifiability under a linear measurement operator.

For gridded VLBI, the Fisher diagonal computed here is the naturally weighted
sampling function described in Thompson, Moran, and Swenson,
*Interferometry and Synthesis in Radio Astronomy*, Chapter 5, and Cornwell
et al. 2008, https://arxiv.org/abs/astro-ph/0701171. The additive perturbation
uses the post-hoc parameterization idea of BayesRays, but the resulting
operator diagnostic is a classical interferometric quantity rather than
neural-network uncertainty.

See `docs/fourier_information_derivation.md` for the full derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..forward import DenseLinearOperator, LinearInverseProblem

DEFAULT_BLIND_INFORMATION_FRACTION = 1e-3


@dataclass(frozen=True)
class FourierDataBlindResult:
    """
    Parameters:
    -----------
    information: np.ndarray
        Exact diagonal of the Fourier-coefficient Fisher matrix in unshifted
        FFT ordering.
    variance: np.ndarray
        Elementwise inverse of the Fisher diagonal plus isotropic prior
        precision. This equals marginal posterior variance only when the full
        posterior precision is diagonal.
    blind_mask: np.ndarray
        Boolean mask selecting modes below the information threshold.
    blind_fraction: float
        Fraction of Fourier modes selected by `blind_mask`.
    prior_precision: float
        Scalar defining the isotropic prior precision matrix `alpha * I`.
    """
    information: np.ndarray
    variance: np.ndarray
    blind_mask: np.ndarray
    blind_fraction: float
    prior_precision: float


def fourier_information(
    forward_model: LinearInverseProblem,
    *,
    row_chunk_size: int = 256,
) -> np.ndarray:
    """
    Compute the Fisher information diagonal for additive image Fourier coefficients.

    Let `x_ref` be a fixed image and let `c` contain Fourier coefficients of
    an additive correction. The corrected image and its predicted measurements
    are

    `x(c) = x_ref + inverse_DFT @ c`

    `y_hat(c) = A @ x_ref + (A @ inverse_DFT) @ c`.

    Therefore, the measurement derivative with respect to `c` is
    `B = A @ inverse_DFT`. This function returns its Fisher diagonal
    `diag(B.H @ noise_precision @ B)`. It does not consider off diagonal
    coupling between Fourier modes.

    Parameters:
    -----------
    forward_model: LinearInverseProblem
        Linear measurement matrix and per-measurement noise.
    row_chunk_size: int
        Number of measurement rows transformed together to limit memory use.

    Returns:
    --------
    np.ndarray
        Information map with shape `forward_model.image_shape` in unshifted
        FFT ordering.
    """
    if row_chunk_size <= 0:
        raise ValueError("`row_chunk_size` must be positive.")
    if not isinstance(forward_model.operator, DenseLinearOperator):
        raise TypeError(
            "Fourier information currently requires a dense linear operator."
        )

    image_shape = forward_model.image_shape
    information = np.zeros(image_shape, dtype=np.float64)
    noise = forward_model.noise_standard_deviation

    # Compute diag(H_c), where B = A @ inverse_DFT and
    # H_c = B.H @ diag(sigma^-2) @ B.
    for start in range(
        0,
        forward_model.number_of_measurements,
        row_chunk_size,
    ):
        stop = min(
            start + row_chunk_size,
            forward_model.number_of_measurements,
        )

        # Reshape B measurement rows from (chunk, pixels) to
        # (chunk, height, width) so the inverse DFT acts on the image axes.
        matrix_rows = forward_model.operator.matrix[start:stop].reshape(
            -1,
            *image_shape,
        )
        transformed_rows = np.fft.ifft2(
            matrix_rows.astype(np.complex128),
            axes=(-2, -1),
        )

        # Entry (m, k) contributes |B[m, k]|^2 / sigma[m]^2. The singleton
        # image axes apply one measurement weight to every mode in that row.
        inverse_noise_variance = (
            1.0 / noise[start:stop, None, None] ** 2
        )

        # Summing the measurement axis gives Fisher diagonal entry
        # I[k] = sum_m |B[m, k]|^2 / sigma[m]^2 for every Fourier mode.
        information += np.sum(
            np.abs(transformed_rows) ** 2 * inverse_noise_variance,
            axis=0,
        )

    # A real image requires c[-k] = conj(c[k]), so a mode and its negative
    # frequency cannot vary independently. In unshifted FFT indexing, flip
    # then roll maps k to (-k) modulo each image dimension. Both entries give
    # the pooled information 0.5 * (I[k] + I[-k]).
    reflected_information = np.roll(
        np.flip(information, axis=(0, 1)),
        shift=1,
        axis=(0, 1),
    )
    return 0.5 * (information + reflected_information)


def fourier_data_blind_map(
    forward_model: LinearInverseProblem,
    *,
    prior_precision_fraction: float = 1e-6,
    blind_information_fraction: float = DEFAULT_BLIND_INFORMATION_FRACTION,
    row_chunk_size: int = 256,
) -> FourierDataBlindResult:
    """
    Identify Fourier modes that receive negligible information from the data.

    This diagnostic adds Fourier perturbations after image reconstruction, so
    it measures the forward operator rather than uncertainty in network
    parameters. Let `I_k` be the Fisher diagonal from `fourier_information`.
    The function sets `alpha = prior_precision_fraction * max(I)` and returns
    the diagonal approximation `1 / (I_k + alpha)`. Modes below
    `blind_information_fraction * max(I)` are marked data blind.

    Parameters:
    -----------
    forward_model: LinearInverseProblem
        Linear measurement system.
    prior_precision_fraction: float
        Isotropic prior precision relative to maximum data information. This
        is a regularization heuristic rather than a calibrated Fourier prior.
    blind_information_fraction: float
        Modes below this fraction of maximum information are marked blind.
    row_chunk_size: int
        Measurement rows processed at once.

    Returns:
    --------
    FourierDataBlindResult
        Per-mode information, variance, blind mask, and threshold summary.
    """
    if prior_precision_fraction < 0.0:
        raise ValueError("`prior_precision_fraction` must be non-negative.")
    if not 0.0 <= blind_information_fraction <= 1.0:
        raise ValueError("`blind_information_fraction` must be in [0, 1].")

    information = fourier_information(forward_model, row_chunk_size=row_chunk_size)
    maximum_information = float(information.max())

    # Use isotropic prior precision Lambda_0 = alpha I, with
    # alpha = prior_precision_fraction * max_k information[k].
    prior_precision = prior_precision_fraction * maximum_information
    if maximum_information == 0.0 and prior_precision == 0.0:
        raise ValueError(
            "The forward model has zero Fourier information and zero prior."
        )

    # Approximate diag((H_c + alpha I)^-1) by
    # 1 / (diag(H_c) + alpha). note that equality requires diagonal H_c.
    variance = 1.0 / (information + prior_precision)

    # Blindness depends only on relative data information, not on the prior.
    blind_mask = information < blind_information_fraction * maximum_information

    return FourierDataBlindResult(
        information=information,
        variance=variance,
        blind_mask=blind_mask,
        blind_fraction=float(blind_mask.mean()),
        prior_precision=prior_precision,
    )
