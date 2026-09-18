"""Parallel-beam CT projection operators from the original 2D CT notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import ndimage as jax_ndimage
from jax.typing import ArrayLike
from skimage.transform import resize_local_mean

from ..image_loading import load_scalar_image
from .linear_problem import LinearInverseProblem

type InterpolationOrder = Literal[0, 1]

SUPPORTED_INTERPOLATION_ORDERS: tuple[InterpolationOrder, ...] = (0, 1)


@dataclass(frozen=True)
class RadonProblemConfig:
    """
    Store model-cache configuration shared by CT training and evaluation.

    Parameters:
    -----------
    source_image_path: Path
        Scalar source image used to generate synthetic CT projections.
    source_array_key: str | None
        Lookup key for an NPZ source. Required for multi-array archives.
    pixel_count: int
        Number of pixels along each square reconstruction-image axis.
    number_of_projection_angles: int
        Uniform projection angles sampled over `[0, pi)`.
    interpolation_order: InterpolationOrder
        Zero for nearest-neighbor rotation or one for bilinear rotation.
    noise_standard_deviation: float
        Standard deviation assigned to every projection measurement.
    """

    source_image_path: Path
    source_array_key: str | None
    pixel_count: int
    number_of_projection_angles: int
    interpolation_order: InterpolationOrder
    noise_standard_deviation: float

    @property
    def input_paths(self) -> dict[str, Path]:
        """Source files hashed into model provenance."""
        return {"source_image": self.source_image_path}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RadonProblemConfig:
        """Restore typed problem configuration from saved JSON metadata."""
        restored_values = dict(values)
        restored_values["source_image_path"] = Path(
            restored_values["source_image_path"]
        )
        return cls(**restored_values)


def _rotated_sample_coordinates(
    image_shape: tuple[int, int],
    angle_radians: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-image row and column coordinates for a rotated output grid."""
    height, width = image_shape
    row_axis = np.arange(height, dtype=np.float32) / height - 0.5
    column_axis = np.arange(width, dtype=np.float32) / width - 0.5
    rows, columns = np.meshgrid(row_axis, column_axis, indexing="ij")

    rotated_columns = (
        columns * np.cos(angle_radians) - rows * np.sin(angle_radians)
    )
    rotated_rows = (
        columns * np.sin(angle_radians) + rows * np.cos(angle_radians)
    )
    source_rows = (rotated_rows + 0.5) * height
    source_columns = (rotated_columns + 0.5) * width
    return source_rows, source_columns


def _rotated_sample_coordinates_jax(
    image_shape: tuple[int, int],
    angle_radians: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """JAX equivalent of `_rotated_sample_coordinates` for differentiation."""
    height, width = image_shape
    coordinate_dtype = angle_radians.dtype
    row_axis = jnp.arange(height, dtype=coordinate_dtype) / height - 0.5
    column_axis = jnp.arange(width, dtype=coordinate_dtype) / width - 0.5
    rows, columns = jnp.meshgrid(row_axis, column_axis, indexing="ij")

    rotated_columns = (
        columns * jnp.cos(angle_radians) - rows * jnp.sin(angle_radians)
    )
    rotated_rows = (
        columns * jnp.sin(angle_radians) + rows * jnp.cos(angle_radians)
    )
    source_rows = (rotated_rows + 0.5) * height
    source_columns = (rotated_columns + 0.5) * width
    return source_rows, source_columns


def radon_transform(
    image: jax.Array,
    projection_angles_radians: jax.Array,
    *,
    interpolation_order: InterpolationOrder = 0,
) -> jax.Array:
    """
    Project an image at multiple parallel-beam CT angles.

    This is the rotate-and-average operator from Tancik et al.'s 2D CT
    notebook. The original implementation uses nearest-neighbor interpolation
    (`interpolation_order=0`). Bilinear interpolation is available explicitly.

    Parameters:
    -----------
    image: jax.Array
        Two-dimensional image with shape `(height, width)`.
    projection_angles_radians: jax.Array
        One-dimensional array of projection angles in radians.
    interpolation_order: int
        Interpolation order passed to `jax.scipy.ndimage.map_coordinates`.
        Supported values are zero for nearest neighbor and one for bilinear.

    Returns:
    --------
    jax.Array
        Sinogram with shape `(number_of_angles, width)`. Each row is the mean
        of the rotated image along its row axis.

    Raises:
    -------
    ValueError
        If the image is not two-dimensional or the interpolation order is not
        supported.
    """
    if image.ndim != 2:
        raise ValueError("`image` must be two-dimensional.")
    if interpolation_order not in SUPPORTED_INTERPOLATION_ORDERS:
        raise ValueError(
            f"`interpolation_order` must be one of "
            f"{SUPPORTED_INTERPOLATION_ORDERS}."
        )

    image_shape = (int(image.shape[0]), int(image.shape[1]))

    def project_one_angle(angle_radians: jax.Array) -> jax.Array:
        source_rows, source_columns = _rotated_sample_coordinates_jax(
            image_shape,
            angle_radians,
        )

        sample_coordinates = jnp.stack(
            [source_rows, source_columns],
            axis=0,
        )
        rotated_image = jax_ndimage.map_coordinates(
            image,
            sample_coordinates,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
        )
        return rotated_image.reshape(image_shape).mean(axis=0)
    return jax.vmap(project_one_angle)(projection_angles_radians)


@dataclass(frozen=True)
class RadonOperator:
    """
    Apply CT projections as an implicit linear operator.

    Projections are computed with `radon_transform`--a full dense Radon matrix
    is never formed or stored.

    Parameters:
    -----------
    image_shape: tuple[int, int]
        `(height, width)` of accepted images.
    projection_angles_radians: np.ndarray
        One-dimensional projection angles in radians.
    interpolation_order: InterpolationOrder
        Zero for nearest-neighbor sampling or one for linear interpolation.
    """

    image_shape: tuple[int, int]
    projection_angles_radians: np.ndarray
    interpolation_order: InterpolationOrder = 0

    def __post_init__(self) -> None:
        """Normalize and validate operator settings."""
        if len(self.image_shape) != 2 or any(
            dimension <= 0 for dimension in self.image_shape
        ):
            raise ValueError("`image_shape` must contain two positive dimensions.")
        angles = np.asarray(self.projection_angles_radians, dtype=np.float32)
        if angles.ndim != 1 or angles.size == 0:
            raise ValueError(
                "`projection_angles_radians` must be a non-empty vector."
            )
        if self.interpolation_order not in SUPPORTED_INTERPOLATION_ORDERS:
            raise ValueError(
                f"`interpolation_order` must be one of {SUPPORTED_INTERPOLATION_ORDERS}."
            )
        object.__setattr__(self, "projection_angles_radians", angles)

    @property
    def measurement_shape(self) -> tuple[int, int]:
        """Sinogram has shape `(number_of_angles, detector_count)`."""
        return (self.projection_angles_radians.size, self.image_shape[1])

    def apply(
        self,
        image: ArrayLike,
        measurement_indices: ArrayLike | None = None,
    ) -> jax.Array:
        """
        Project one image.

        Parameters:
        -----------
        image: ArrayLike
            Image with shape `(H, W)` or the equivalent vector `(H * W,)`.
        measurement_indices: ArrayLike | None
            Optional indices selected after computing the complete sinogram.

        Returns:
        --------
        jax.Array
            Flattened sinogram with shape `(T * W,)`, where `T` is projection
            angle count.
        """
        image_array = jnp.asarray(image)
        if image_array.size != int(np.prod(self.image_shape)):
            raise ValueError(
                f"`image` contains {image_array.size} values; expected "
                f"{int(np.prod(self.image_shape))}."
            )
        sinogram = radon_transform(
            image_array.reshape(self.image_shape),
            jnp.asarray(self.projection_angles_radians),
            interpolation_order=self.interpolation_order,
        )
        flattened_sinogram = sinogram.reshape(-1)
        if measurement_indices is not None:
            return flattened_sinogram[jnp.asarray(measurement_indices)]
        return flattened_sinogram

    def apply_to_columns(self, image_columns: ArrayLike) -> jax.Array:
        """
        Project multiple flattened images or image perturbations.

        Parameters:
        -----------
        image_columns: ArrayLike
            Matrix with shape `(H * W, P)`. Each column is independently
            reshaped to `(H, W)` and passed through `radon_transform`.

        Returns:
        --------
        jax.Array
            Measurement matrix with shape `(T * W, P)`. Column `p` contains
            the flattened sinogram produced by image column `p`.
        """
        columns = jnp.asarray(image_columns)
        expected_rows = int(np.prod(self.image_shape))
        if columns.ndim != 2 or columns.shape[0] != expected_rows:
            raise ValueError(
                "`image_columns` must have shape "
                "(number_of_pixels, number_of_columns)."
            )
        return jax.vmap(self.apply, in_axes=1, out_axes=1)(columns)


def build_radon_matrix(
    image_shape: tuple[int, int],
    projection_angles_radians: np.ndarray,
    *,
    interpolation_order: InterpolationOrder = 0,
) -> np.ndarray:
    """
    Materialize the CT projection as a matrix acting on flattened images.

    Parameters:
    -----------
    image_shape: tuple[int, int]
        `(height, width)` of the input image.
    projection_angles_radians: np.ndarray
        One-dimensional array of projection angles in radians.
    interpolation_order: int
        Pass zero for nearest-neighbor or one for bilinear interpolation.

    Returns:
    --------
    np.ndarray
        Matrix with shape
        `(number_of_angles * width, height * width)`. Multiplying this matrix
        by a row-major flattened image produces a flattened sinogram.

    Raises:
    -------
    ValueError
        If dimensions, angles, or interpolation order are invalid.
    """
    if len(image_shape) != 2 or any(dimension <= 0 for dimension in image_shape):
        raise ValueError("`image_shape` must contain two positive dimensions.")
    if interpolation_order not in SUPPORTED_INTERPOLATION_ORDERS:
        raise ValueError(
            f"`interpolation_order` must be one of "
            f"{SUPPORTED_INTERPOLATION_ORDERS}."
        )

    angles = np.asarray(projection_angles_radians, dtype=np.float32)
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("`projection_angles_radians` must be a non-empty vector.")

    height, width = image_shape
    number_of_pixels = height * width
    matrix = np.zeros(
        (angles.size * width, number_of_pixels),
        dtype=np.float32,
    )
    
    # Every output sample at image position (row, column) contributes to the
    # detector bin identified by its column.
    detector_bins = np.broadcast_to(
        np.arange(width)[None, :],
        image_shape,
    )

    for angle_index, angle_radians in enumerate(angles):
        source_rows, source_columns = _rotated_sample_coordinates(
            image_shape,
            angle_radians,
        )
        # Flatten measurement coordinates (angle, detector bin) into matrix rows
        output_rows = angle_index * width + detector_bins

        if interpolation_order == 0:
            # select the nearest integer source pixel
            nearest_rows = np.floor(source_rows + 0.5).astype(int)
            nearest_columns = np.floor(source_columns + 0.5).astype(int)
            inside = (
                (nearest_rows >= 0)
                & (nearest_rows < height)
                & (nearest_columns >= 0)
                & (nearest_columns < width)
            )
            source_pixels = (
                nearest_rows[inside] * width + nearest_columns[inside]
            )
            # Each detector value is the mean of H rotated samples.
            # Repeated source-pixel hits must accumulate rather than overwrite so we use 
            # np.add.at
            np.add.at(
                matrix,
                (output_rows[inside], source_pixels),
                1.0 / height,
            )
            continue

        # bilinear interpolation
        lower_rows = np.floor(source_rows).astype(int)
        lower_columns = np.floor(source_columns).astype(int)
        row_fraction = source_rows - lower_rows
        column_fraction = source_columns - lower_columns
        corners = (
            ((0, 0), (1.0 - row_fraction) * (1.0 - column_fraction)),
            ((1, 0), row_fraction * (1.0 - column_fraction)),
            ((0, 1), (1.0 - row_fraction) * column_fraction),
            ((1, 1), row_fraction * column_fraction),
        )
        for (row_offset, column_offset), interpolation_weight in corners:
            source_row = lower_rows + row_offset
            source_column = lower_columns + column_offset
            inside = (
                (source_row >= 0)
                & (source_row < height)
                & (source_column >= 0)
                & (source_column < width)
            )
            source_pixels = source_row[inside] * width + source_column[inside]
            np.add.at(
                matrix,
                (output_rows[inside], source_pixels),
                interpolation_weight[inside] / height,
            )
    return matrix


def build_radon_inverse_problem(
    source_image: np.ndarray,
    projection_angles_radians: np.ndarray,
    *,
    noise_standard_deviation: float | np.ndarray,
    interpolation_order: InterpolationOrder = 0,
) -> LinearInverseProblem:
    """
    Construct a noiseless linear CT dataset from a source image.

    Parameters:
    -----------
    source_image: np.ndarray
        Two-dimensional ground-truth image.
    projection_angles_radians: np.ndarray
        Projection angles in radians.
    noise_standard_deviation: float | np.ndarray
        Assumed measurement noise. A scalar is broadcast to all sinogram
        values; an array must contain one value per measurement.
    interpolation_order: int
        Interpolation convention used by the projection matrix.

    Returns:
    --------
    LinearInverseProblem
        Implicit Radon operator, synthetic noiseless sinogram, noise values,
        and image dimensions.
    """
    image = np.asarray(source_image)
    if image.ndim != 2:
        raise ValueError("`source_image` must be two-dimensional.")

    operator = RadonOperator(
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        projection_angles_radians=projection_angles_radians,
        interpolation_order=interpolation_order,
    )
    observations = np.asarray(operator.apply(image))
    noise = np.asarray(noise_standard_deviation, dtype=float)
    if noise.ndim == 0:
        noise = np.full(observations.size, float(noise))

    return LinearInverseProblem(
        operator=operator,
        observed_measurements=observations,
        noise_standard_deviation=noise,
        description=(
            f"Parallel-beam CT with {len(projection_angles_radians)} angles "
            f"and interpolation order {interpolation_order}"
        ),
    )


def build_radon_problem_from_config(
    config: RadonProblemConfig,
) -> tuple[LinearInverseProblem, np.ndarray]:
    """
    Rebuild a CT problem and normalized truth from its saved recipe.

    Parameters:
    -----------
    config: RadonProblemConfig
        Source, resolution, projection, interpolation, and noise settings.

    Returns:
    --------
    problem: LinearInverseProblem
        Synthetic CT measurements and implicit Radon operator.
    truth: np.ndarray
        Resized source image normalized by its maximum intensity.
    """
    source_image = load_scalar_image(
        config.source_image_path,
        array_key=config.source_array_key,
    )
    resized_image = resize_local_mean(
        source_image,
        (config.pixel_count, config.pixel_count),
        grid_mode=True,
        preserve_range=True,
    )
    maximum_intensity = float(resized_image.max())
    if maximum_intensity <= 0.0:
        raise ValueError("The resized source image must have positive intensity.")
    truth = resized_image / maximum_intensity
    projection_angles = np.linspace(
        0.0,
        np.pi,
        config.number_of_projection_angles,
        endpoint=False,
        dtype=np.float32,
    )
    problem = build_radon_inverse_problem(
        truth,
        projection_angles,
        noise_standard_deviation=config.noise_standard_deviation,
        interpolation_order=config.interpolation_order,
    )
    return problem, truth
