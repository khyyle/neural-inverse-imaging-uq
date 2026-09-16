"""Shared linear inverse problem for inverse-imaging experiments."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .operator import LinearOperator


@dataclass(frozen=True)
class LinearInverseProblem:
    """
    Describe observations produced by a fixed linear measurement operator.

    The problem represents

    ``observed_measurements = operator.apply(image) + noise``.

    Parameters:
    -----------
    operator: LinearOperator
        Dense or implicit mapping from an image into measurement space.
    observed_measurements: np.ndarray
        Measured values with shape `(number_of_measurements,)`.
    noise_standard_deviation: np.ndarray
        Positive standard deviation for each measurement, shape
        `(number_of_measurements,)`.
    field_of_view_microarcseconds: float | None
        Angular field of view for astronomical images. `None` for domains
        without an angular scale.
    description: str
        Human-readable label included in experiment provenance.
    """

    operator: LinearOperator
    observed_measurements: np.ndarray
    noise_standard_deviation: np.ndarray
    field_of_view_microarcseconds: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Normalize measurements and reject incompatible dimensions."""
        observed_measurements = np.asarray(self.observed_measurements)
        noise_standard_deviation = np.asarray(
            self.noise_standard_deviation,
            dtype=float,
        )

        if observed_measurements.ndim != 1:
            raise ValueError("`observed_measurements` must be one-dimensional.")
        if noise_standard_deviation.ndim != 1:
            raise ValueError("`noise_standard_deviation` must be one-dimensional.")

        number_of_measurements = int(np.prod(self.operator.measurement_shape))
        if observed_measurements.shape[0] != number_of_measurements:
            raise ValueError(
                "`observed_measurements` length must match the operator output size."
            )
        if noise_standard_deviation.shape[0] != number_of_measurements:
            raise ValueError(
                "`noise_standard_deviation` length must match the operator output size."
            )
        if not np.all(np.isfinite(noise_standard_deviation)):
            raise ValueError("`noise_standard_deviation` must contain finite values.")
        if np.any(noise_standard_deviation <= 0.0):
            raise ValueError("`noise_standard_deviation` must be strictly positive.")

        if self.field_of_view_microarcseconds is not None and self.field_of_view_microarcseconds <= 0.0:
            raise ValueError("`field_of_view_microarcseconds` must be positive.")

        object.__setattr__(self, "observed_measurements", observed_measurements)
        object.__setattr__(self, "noise_standard_deviation", noise_standard_deviation)

    @property
    def number_of_measurements(self) -> int:
        """Number of scalar or complex measurements represented by the operator."""
        return int(np.prod(self.operator.measurement_shape))

    @property
    def number_of_pixels(self) -> int:
        """Number of pixels in the flattened image domain."""
        return int(np.prod(self.operator.image_shape))

    @property
    def image_shape(self) -> tuple[int, int]:
        """Two-dimensional image shape accepted by the operator."""
        return self.operator.image_shape

    def predict(
        self,
        image: ArrayLike,
        measurement_indices: ArrayLike | None = None,
    ) -> jax.Array:
        """
        Apply the fixed measurement operator to an image.

        Parameters:
        -----------
        image: ArrayLike
            Image with shape `image_shape`, or an already flattened vector with
            `number_of_pixels` entries.
        measurement_indices: ArrayLike | None
            Optional flattened measurement indices to evaluate.

        Returns:
        --------
        jax.Array
            All predicted measurements, or only the requested subset.
        """
        return self.operator.apply(image, measurement_indices)

    def gaussian_negative_log_likelihood(
        self,
        image: ArrayLike,
        measurement_indices: ArrayLike | None = None,
    ) -> jax.Array:
        """
        Evaluate the Gaussian data term up to additive constants.

        Parameters:
        -----------
        image: ArrayLike
            Image accepted by `predict`.
        measurement_indices: ArrayLike | None
            Optional flattened measurement indices to evaluate.

        Returns:
        --------
        jax.Array
            Scalar `0.5 * mean(abs((prediction - observation) / sigma) ** 2)`.

        Notes:
        ------
        For complex measurements, `sigma` is the standard deviation of each
        independent real and imaginary component.
        """
        predictions = self.predict(image, measurement_indices)
        observations = jnp.asarray(self.observed_measurements)
        noise = jnp.asarray(self.noise_standard_deviation)
        if measurement_indices is not None:
            indices = jnp.asarray(measurement_indices)
            observations = observations[indices]
            noise = noise[indices]
        normalized_residuals = (predictions - observations) / noise
        return 0.5 * jnp.mean(jnp.abs(normalized_residuals) ** 2)

    def apply_to_columns(self, image_columns: ArrayLike) -> jax.Array:
        """
        Apply the operator to image-vector columns.

        Parameters:
        -----------
        image_columns: ArrayLike
            Matrix with shape `(number_of_pixels, number_of_columns)`.

        Returns:
        --------
        jax.Array
            Matrix with shape
            `(number_of_measurements, number_of_columns)`.
        """
        return self.operator.apply_to_columns(image_columns)
