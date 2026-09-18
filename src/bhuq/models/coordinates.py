"""Coordinate grids used to evaluate two-dimensional neural images."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

type CoordinateConvention = Literal[
    "xy_exclusive",
    "row_column_inclusive",
]

SUPPORTED_COORDINATE_CONVENTIONS: tuple[CoordinateConvention, ...] = (
    "xy_exclusive",
    "row_column_inclusive",
)


def build_coordinate_grid(
    image_shape: tuple[int, int],
    convention: CoordinateConvention,
    *,
    flatten: bool = True,
) -> jax.Array:
    """
    Build a normalized coordinate grid under an explicit convention.

    Parameters:
    -----------
    image_shape: tuple[int, int]
        `(height, width)` of the image grid.
    convention: CoordinateConvention
        `xy_exclusive` stores horizontal then vertical coordinates and excludes
        one. `row_column_inclusive` stores row then column coordinates and
        includes both zero and one.
    flatten: bool
        Whether to combine the image axes in row-major order.

    Returns:
    --------
    jax.Array
        Coordinate array with shape `(height * width, 2)` when flattened or
        `(height, width, 2)` otherwise.

    Raises:
    -------
    ValueError
        If the image shape or convention is invalid.
    """
    if len(image_shape) != 2 or any(
        dimension <= 0 for dimension in image_shape
    ):
        raise ValueError("`image_shape` must contain two positive dimensions.")
    if convention not in SUPPORTED_COORDINATE_CONVENTIONS:
        raise ValueError(
            f"`convention` must be one of "
            f"{SUPPORTED_COORDINATE_CONVENTIONS}."
        )

    height, width = image_shape
    if convention == "xy_exclusive":
        horizontal_axis = np.arange(width) / width
        vertical_axis = np.arange(height) / height
        horizontal, vertical = np.meshgrid(
            horizontal_axis,
            vertical_axis,
            indexing="xy",
        )
        coordinates = np.stack((horizontal, vertical), axis=-1)
    else:
        row_axis = np.linspace(0.0, 1.0, height)
        column_axis = np.linspace(0.0, 1.0, width)
        rows, columns = np.meshgrid(
            row_axis,
            column_axis,
            indexing="ij",
        )
        coordinates = np.stack((rows, columns), axis=-1)

    if flatten:
        coordinates = coordinates.reshape(-1, 2)
    return jnp.asarray(coordinates)
