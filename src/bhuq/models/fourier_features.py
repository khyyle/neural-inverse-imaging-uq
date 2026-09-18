"""
Fourier-feature MLP used by the CT experiment.

The encoding and architecture follow the 2D CT experiment from:
https://github.com/tancik/fourier-feature-networks

The original notebook uses `jax.example_libraries.stax.Dense` without
initializer arguments. The network here is implemented using flax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

type ParameterTree = Any


@dataclass(frozen=True)
class FourierFeatureModelConfig:
    """Store the model-cache recipe shared by CT training and evaluation."""

    number_of_frequencies: int
    frequency_scale: float
    frequency_seed: int
    network_depth: int
    network_width: int

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FourierFeatureModelConfig:
        """Restore typed model configuration from saved JSON metadata."""
        return cls(**values)


def fourier_features(
    coordinates: jax.Array,
    frequency_matrix: jax.Array,
) -> jax.Array:
    """
    Encode coordinates with a supplied Fourier frequency matrix.

    Parameters:
    -----------
    coordinates: jax.Array
        Coordinates with shape `(..., number_of_dimensions)`.
    frequency_matrix: jax.Array
        Frequencies with shape
        `(number_of_frequencies, number_of_dimensions)`.

    Returns:
    --------
    jax.Array
        Concatenated sine and cosine features with shape
        `(..., 2 * number_of_frequencies)`.
    """
    # (..., D) @ (D, F) -> (..., F)
    arguments = 2.0 * jnp.pi * coordinates @ frequency_matrix.T
    return jnp.concatenate(
        [jnp.sin(arguments), jnp.cos(arguments)],
        axis=-1,
    )


def sample_gaussian_frequencies(
    random_key: jax.Array,
    *,
    number_of_frequencies: int = 256,
    coordinate_dimensions: int = 2,
    scale: float = 4.0,
) -> jax.Array:
    """
    Sample a Gaussian Fourier frequency matrix.

    Parameters:
    -----------
    random_key: jax.Array
        JAX random key.
    number_of_frequencies: int
        Number of frequency vectors.
    coordinate_dimensions: int
        Number of coordinate dimensions.
    scale: float
        Standard deviation of sampled frequency entries.

    Returns:
    --------
    jax.Array
        Matrix with shape
        `(number_of_frequencies, coordinate_dimensions)`.
    """
    if number_of_frequencies <= 0:
        raise ValueError("`number_of_frequencies` must be positive.")
    if coordinate_dimensions <= 0:
        raise ValueError("`coordinate_dimensions` must be positive.")
    if scale <= 0.0:
        raise ValueError("`scale` must be positive.")
    return scale * jax.random.normal(
        random_key,
        (number_of_frequencies, coordinate_dimensions),
    )


def build_fourier_feature_coordinate_grid(
    image_shape: tuple[int, int],
    *,
    flatten: bool = True,
) -> jax.Array:
    """
    Build the exclusive horizontal-vertical grid used by ct models.

    Parameters:
    -----------
    image_shape: tuple[int, int]
        `(height, width)` of the image grid.
    flatten: bool
        If `True`, combine the image axes in row-major order. If `False`,
        retain the two image dimensions.

    Returns:
    --------
    jax.Array
        Shape `(height * width, 2)` when flattened, otherwise
        `(height, width, 2)`. Both axes exclude one.

    Notes:
    ------
    This uses `meshgrid(..., indexing="xy")` so both normalized axes include zero and
    exclude one. For a dimension of size `n`, spacing is `1 / n` with final
    coordinate `(n - 1) / n`. This is equivalent to `linspace(0, 1, n, endpoint=False)`. 
    Follows convention used in the Tancik CT experiment.
    """
    if len(image_shape) != 2 or any(dimension <= 0 for dimension in image_shape):
        raise ValueError("`image_shape` must contain two positive dimensions.")

    height, width = image_shape
    horizontal_axis = np.arange(width) / width
    vertical_axis = np.arange(height) / height
    horizontal, vertical = np.meshgrid(
        horizontal_axis,
        vertical_axis,
        indexing="xy",
    )
    coordinates = np.stack([horizontal, vertical], axis=-1)
    if flatten:
        coordinates = coordinates.reshape(-1, 2)
    return jnp.asarray(coordinates)


class FourierFeatureMLP(nn.Module):
    """
    Render scalar values from Fourier-encoded coordinates.

    Parameters:
    -----------
    frequency_matrix: jax.Array
        Fixed frequencies with shape `(number_of_frequencies, dimensions)`.
    network_depth: int
        Total number of dense layers, including the output layer.
    network_width: int
        Width of each hidden dense layer.
    """

    frequency_matrix: jax.Array
    network_depth: int = 4
    network_width: int = 256

    @nn.compact
    def __call__(
        self,
        coordinates: jax.Array,
        *,
        return_last_layer_inputs: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
        """
        Evaluate the Fourier-feature MLP.

        Parameters:
        -----------
        coordinates: jax.Array
            Coordinates with shape `(..., number_of_dimensions)`.
        return_last_layer_inputs: bool
            Whether to return output-layer inputs and logits with the output.

        Returns:
        --------
        jax.Array | tuple[jax.Array, jax.Array, jax.Array]
            Scalar sigmoid output, optionally with output-layer inputs and
            logits.
        """
        if self.network_depth < 2:
            raise ValueError("`network_depth` must be at least two.")
        if self.network_width <= 0:
            raise ValueError("`network_width` must be positive.")

        hidden_values = fourier_features(
            coordinates,
            self.frequency_matrix,
        )
        for _ in range(self.network_depth - 1):
            # These explicitly match the stax.Dense defaults used by Tancik et al. 2020
            hidden_values = nn.Dense(
                self.network_width,
                kernel_init=jax.nn.initializers.glorot_normal(),
                bias_init=jax.nn.initializers.normal(),
            )(hidden_values)
            hidden_values = nn.relu(hidden_values)

        last_layer_inputs = hidden_values
        logits = nn.Dense(
            1,
            kernel_init=jax.nn.initializers.glorot_normal(),
            bias_init=jax.nn.initializers.normal(),
        )(last_layer_inputs)[..., 0]
        output = nn.sigmoid(logits)
        if return_last_layer_inputs:
            return output, last_layer_inputs, logits
        return output


def last_layer_image_jacobian(
    model: FourierFeatureMLP,
    parameters: ParameterTree,
    coordinates: jax.Array,
) -> np.ndarray:
    """
    Differentiate outputs with respect to final-layer weights and bias.

    Parameters:
    -----------
    model: FourierFeatureMLP
        Fitted Fourier-feature model.
    parameters: ParameterTree
        Flax parameter tree.
    coordinates: jax.Array
        Coordinates with shape `(number_of_points, dimensions)`.

    Returns:
    --------
    np.ndarray
        Jacobian with one row per coordinate and one column per final-layer
        weight and bias.
    """
    _output, last_layer_inputs, logits = model.apply(
        {"params": parameters},
        coordinates,
        return_last_layer_inputs=True,
    )
    output = jax.nn.sigmoid(logits)
    sigmoid_derivative = output * (1.0 - output)
    last_layer_inputs_with_bias = jnp.concatenate(
        [
            last_layer_inputs,
            jnp.ones(
                (last_layer_inputs.shape[0], 1),
                dtype=last_layer_inputs.dtype,
            ),
        ],
        axis=1,
    )

    # (N, 1) * (N, F + 1) -> (N, F + 1)
    return np.asarray(
        sigmoid_derivative[:, None] * last_layer_inputs_with_bias
    )
