"""Centered Fourier-crop measurements from the original Fourier notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from skimage.transform import resize_local_mean

from ..image_loading import load_scalar_image, normalize_image_by_maximum
from .linear_problem import LinearInverseProblem


def centered_fourier_crop(
    image: ArrayLike,
    crop_size: int,
) -> jax.Array:
    """
    Return a centered square crop of an orthonormal two-dimensional FFT.

    the zero-frequency coefficient is shifted to the image center before taking the `crop_size`
    by `crop_size` region.

    Parameters:
    -----------
    image: ArrayLike
        Two-dimensional scalar image.
    crop_size: int
        Width and height of the centered Fourier region.

    Returns:
    --------
    jax.Array
        Complex Fourier coefficients with shape `(crop_size, crop_size)`.
    """
    image_array = jnp.asarray(image)
    if image_array.ndim != 2:
        raise ValueError("`image` must be two-dimensional.")
    if crop_size <= 0:
        raise ValueError("`crop_size` must be positive.")
    if crop_size > min(image_array.shape):
        raise ValueError("`crop_size` must not exceed either image dimension.")

    shifted_fourier = jnp.fft.fftshift(
        jnp.fft.fft2(image_array, norm="ortho"),
    )
    row_start = (image_array.shape[0] - crop_size) // 2
    column_start = (image_array.shape[1] - crop_size) // 2
    return jax.lax.dynamic_slice(
        shifted_fourier,
        (row_start, column_start),
        (crop_size, crop_size),
    )


@dataclass(frozen=True)
class FourierCropProblemConfig:
    """
    Store the recipe for centered Fourier-crop image reconstruction.

    Parameters:
    -----------
    source_image_path: Path
        Scalar source image used to generate synthetic measurements.
    source_array_key: str | None
        Lookup key for an NPZ source. Required for multi-array archives.
    pixel_count: int
        Number of pixels along each square reconstruction-image axis.
    crop_size: int
        Width and height of the centered Fourier region.
    noise_standard_deviation: float
        Assumed standard deviation of each complex coefficient's independent
        real and imaginary components after image normalization.
    """

    source_image_path: Path
    source_array_key: str | None
    pixel_count: int
    crop_size: int
    noise_standard_deviation: float

    def __post_init__(self) -> None:
        """Reject invalid dimensions and noise before building the problem."""
        if self.pixel_count <= 0:
            raise ValueError("`pixel_count` must be positive.")
        if self.crop_size <= 0 or self.crop_size > self.pixel_count:
            raise ValueError("`crop_size` must be positive and not exceed `pixel_count`.")
        if self.noise_standard_deviation <= 0.0:
            raise ValueError("`noise_standard_deviation` must be positive.")

    @property
    def input_paths(self) -> dict[str, Path]:
        """Source files hashed into model provenance."""
        return {"source_image": self.source_image_path}

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> FourierCropProblemConfig:
        """Restore typed problem configuration from saved JSON metadata."""
        restored_values = dict(values)
        restored_values["source_image_path"] = Path(restored_values["source_image_path"])
        return cls(**restored_values)


@dataclass(frozen=True)
class FourierCropOperator:
    """
    Measure the centered low-frequency region of a two-dimensional FFT.

    Parameters:
    -----------
    image_shape: tuple[int, int]
        `(height, width)` of accepted images.
    crop_size: int
        Width and height of the centered Fourier region.
    """

    image_shape: tuple[int, int]
    crop_size: int

    def __post_init__(self) -> None:
        """Reject invalid image and crop dimensions."""
        if len(self.image_shape) != 2 or any(dimension <= 0 for dimension in self.image_shape):
            raise ValueError("`image_shape` must contain two positive dimensions.")
        if self.crop_size <= 0:
            raise ValueError("`crop_size` must be positive.")
        if self.crop_size > min(self.image_shape):
            raise ValueError("`crop_size` must not exceed either image dimension.")

    @property
    def measurement_shape(self) -> tuple[int, int]:
        """Fourier measurements retain the square crop shape."""
        return self.crop_size, self.crop_size

    def apply(
        self,
        image: ArrayLike,
        measurement_indices: ArrayLike | None = None,
    ) -> jax.Array:
        """
        Transform one image and return flattened complex measurements.

        Parameters:
        -----------
        image: ArrayLike
            Image with `image_shape`, or its row-major flattened vector.
        measurement_indices: ArrayLike | None
            Optional indices selected after flattening the Fourier crop.

        Returns:
        --------
        jax.Array
            Complex vector containing `crop_size**2` coefficients, or the
            selected subset.
        """
        image_array = jnp.asarray(image)
        expected_pixels = int(np.prod(self.image_shape))
        if image_array.size != expected_pixels:
            raise ValueError(
                f"`image` contains {image_array.size} values; expected {expected_pixels}."
            )
        measurements = centered_fourier_crop(
            image_array.reshape(self.image_shape),
            self.crop_size,
        ).reshape(-1)
        if measurement_indices is not None:
            return measurements[jnp.asarray(measurement_indices)]
        return measurements

    def apply_to_columns(self, image_columns: ArrayLike) -> jax.Array:
        """
        Apply the Fourier crop independently to image-vector columns.

        Parameters:
        -----------
        image_columns: ArrayLike
            Matrix with shape `(number_of_pixels, number_of_columns)`.

        Returns:
        --------
        jax.Array
            Matrix with shape
            `(crop_size**2, number_of_columns)`.
        """
        columns = jnp.asarray(image_columns)
        expected_pixels = int(np.prod(self.image_shape))
        if columns.ndim != 2 or columns.shape[0] != expected_pixels:
            raise ValueError(
                "`image_columns` must have shape (number_of_pixels, number_of_columns)."
            )
        return jax.vmap(self.apply, in_axes=1, out_axes=1)(columns)


def build_fourier_crop_inverse_problem(
    source_image: np.ndarray,
    crop_size: int,
    *,
    noise_standard_deviation: float | np.ndarray,
) -> LinearInverseProblem:
    """
    Construct noiseless centered Fourier-crop observations of an image.

    Parameters:
    -----------
    source_image: np.ndarray
        Two-dimensional ground-truth image.
    crop_size: int
        Width and height of the centered Fourier region.
    noise_standard_deviation: float | np.ndarray
        Assumed standard deviation of each complex measurement's independent
        real and imaginary components. A scalar is broadcast to every
        coefficient.

    Returns:
    --------
    LinearInverseProblem
        Fourier-crop operator, complex observations, noise, and image shape.
    """
    image = np.asarray(source_image)
    if image.ndim != 2:
        raise ValueError("`source_image` must be two-dimensional.")

    operator = FourierCropOperator(
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        crop_size=crop_size,
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
            f"Centered {crop_size}x{crop_size} crop of an orthonormal two-dimensional FFT"
        ),
    )


def build_fourier_crop_problem_from_config(
    config: FourierCropProblemConfig,
) -> tuple[LinearInverseProblem, np.ndarray]:
    """
    Rebuild a Fourier-crop problem and truth image from its saved recipe.

    The resized source is normalized by its maximum before generating
    noiseless observations, matching the bounded neural-image output.

    Parameters:
    -----------
    config: FourierCropProblemConfig
        Source, image resolution, crop size, and noise assumptions.

    Returns:
    --------
    problem: LinearInverseProblem
        Synthetic centered Fourier-crop measurements and noise model.
    truth: np.ndarray
        Resized source image with maximum intensity one.
    """
    source_image = load_scalar_image(
        config.source_image_path,
        array_key=config.source_array_key,
    )
    truth = normalize_image_by_maximum(
        resize_local_mean(
            source_image,
            (config.pixel_count, config.pixel_count),
            grid_mode=True,
            preserve_range=True,
        )
    )
    problem = build_fourier_crop_inverse_problem(
        truth,
        config.crop_size,
        noise_standard_deviation=config.noise_standard_deviation,
    )
    return problem, truth
