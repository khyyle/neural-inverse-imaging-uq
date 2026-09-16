"""Linear measurement operators with dense and implicit implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


class LinearOperator(Protocol):
    """Structural interface required by linear training and uncertainty code."""

    @property
    def image_shape(self) -> tuple[int, int]:
        """Shape of an image accepted by the operator."""
        ...

    @property
    def measurement_shape(self) -> tuple[int, ...]:
        """Shape of measurements returned by the operator."""
        ...

    def apply(
        self,
        image: ArrayLike,
        measurement_indices: ArrayLike | None = None,
    ) -> jax.Array:
        """
        Map an image into flattened measurement space.

        Parameters:
        -----------
        image: ArrayLike
            Image with `image_shape`, or the equivalent flattened vector.
        measurement_indices: ArrayLike | None
            Optional flattened measurement indices to evaluate. `None`
            evaluates every measurement.

        Returns:
        --------
        jax.Array
            Flattened measurements containing `prod(measurement_shape)` values.
        """
        ...

    def apply_to_columns(self, image_columns: ArrayLike) -> jax.Array:
        """
        Apply the operator independently to image-vector columns.

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
        ...


@dataclass(frozen=True)
class DenseLinearOperator:
    """
    Apply a linear measurement operator stored as a dense matrix.

    Parameters:
    -----------
    matrix: np.ndarray
        Matrix with shape `(number_of_measurements, number_of_pixels)`.
    image_shape: tuple[int, int]
        Two-dimensional image shape represented by matrix columns.
    """

    matrix: np.ndarray
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix)
        if matrix.ndim != 2:
            raise ValueError("`matrix` must be two-dimensional.")
        if len(self.image_shape) != 2 or any(
            dimension <= 0 for dimension in self.image_shape
        ):
            raise ValueError("`image_shape` must contain two positive dimensions.")
        if int(np.prod(self.image_shape)) != matrix.shape[1]:
            raise ValueError("The product of `image_shape` must equal the matrix column count.")
        
        object.__setattr__(self, "matrix", matrix)

    @property
    def measurement_shape(self) -> tuple[int]:
        """One-dimensional output shape of the dense matrix."""
        return (self.matrix.shape[0],)

    def apply(
        self,
        image: ArrayLike,
        measurement_indices: ArrayLike | None = None,
    ) -> jax.Array:
        """Multiply selected dense-matrix rows by a flattened image."""
        image_array = jnp.asarray(image)
        if image_array.size != self.matrix.shape[1]:
            raise ValueError(
                f"`image` contains {image_array.size} values; expected "
                f"{self.matrix.shape[1]}."
            )
        matrix = jnp.asarray(self.matrix)
        if measurement_indices is not None:
            matrix = matrix[jnp.asarray(measurement_indices)]
        return matrix @ image_array.reshape(-1)

    def apply_to_columns(self, image_columns: ArrayLike) -> jax.Array:
        """Multiply the dense matrix by one or more flattened image columns."""
        columns = jnp.asarray(image_columns)
        if columns.ndim != 2 or columns.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                "`image_columns` must have shape "
                "(number_of_pixels, number_of_columns)."
            )
        return jnp.asarray(self.matrix) @ columns
