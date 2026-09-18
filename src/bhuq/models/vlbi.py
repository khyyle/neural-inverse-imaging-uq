"""Coordinate MLP used by VLBI reconstruction."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

type ParameterTree = Any
type ActivationFunction = Callable[[jax.Array], jax.Array]

SINE_WRAP_PERIOD = 100.0 * math.pi
DEFAULT_OUTPUT_LOGIT_OFFSET = 10.0


@dataclass(frozen=True)
class NeuralImageConfig:
    """
    Store the model-cache recipe shared by VLBI training and evaluation.

    Parameters:
    -----------
    positional_encoding_degree: int
        Number of positional-encoding frequency levels.
    network_depth: int
        Number of hidden MLP layers.
    network_width: int
        Number of outputs in each hidden layer.
    output_logit_offset: float
        Value subtracted before the output sigmoid to initialize a dark image.
    """

    positional_encoding_degree: int
    network_depth: int
    network_width: int
    output_logit_offset: float

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> NeuralImageConfig:
        """Restore typed model configuration from saved JSON metadata."""
        return cls(**values)


def _safe_sin(values: jax.Array) -> jax.Array:
    """Evaluate sin after reducing large arguments to a finite interval."""
    return jnp.sin(values % SINE_WRAP_PERIOD)


def positional_encode(
    coordinates: jax.Array,
    degree: int,
) -> jax.Array:
    """
    Append multi-scale sine and cosine features to input coordinates.

    For each coordinate dimension, it evaluates frequencies `2**level` for levels from
    zero through `degree - 1`. Cosine features are evaluated through the
    identity `cos(x) = sin(x + pi / 2)`.

    Parameters:
    -----------
    coordinates: jax.Array
        Coordinates with shape `(..., number_of_dimensions)`.
    degree: int
        Number of frequency levels. Zero returns `coordinates` unchanged.

    Returns:
    --------
    jax.Array
        Encoded coordinates with shape
        `(..., number_of_dimensions * (1 + 2 * degree))`. Raw coordinates
        appear first, followed by sine and cosine features.

    Raises:
    -------
    ValueError
        If `degree` is negative.
    """
    if degree < 0:
        raise ValueError("`degree` must be non-negative.")
    if degree == 0:
        return coordinates

    frequency_scales = jnp.asarray([2**level for level in range(degree)])

    # (..., 1, D) times (L, 1) broadcasts to (..., L, D), where
    # L is the number of frequency levels and D is coordinate dimension.
    scaled_coordinates = coordinates[..., None, :] * frequency_scales[:, None]
    
    # Fold frequency and coordinate axes together: (..., L, D) -> (..., L * D).
    flattened_scales = jnp.reshape(scaled_coordinates, list(coordinates.shape[:-1]) + [-1])

    fourier_features = jnp.concatenate(
        [
            flattened_scales, # sin arguments
            flattened_scales + 0.5 * jnp.pi, # cos arguments
        ],
        axis=-1,
    )
    fourier_features = _safe_sin(fourier_features)
    return jnp.concatenate([coordinates, fourier_features], axis=-1)


class MLP(nn.Module):
    """
    A simple skip-connected multi-layer perceptron

    Parameters:
    -----------
    network_depth: int
        Number of hidden dense layers.
    network_width: int
        Number of outputs in each hidden dense layer.
    activation: ActivationFunction
        Activation applied after each hidden dense layer.
    output_channels: int
        Number of outputs in the final dense layer.
    use_skip_connections: bool
        Whether to concatenate the original encoded coordinates back into the
        hidden representation halfway through the network.
    """

    network_depth: int = 4
    network_width: int = 128
    activation: ActivationFunction = nn.relu
    output_channels: int = 1
    use_skip_connections: bool = True

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        return_last_layer_inputs: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """
        Evaluate the MLP for a batch of encoded coordinates.

        Parameters:
        -----------
        x: jax.Array
            Encoded coordinates with shape `(..., number_of_features)`.

        Returns:
        --------
        jax.Array | tuple[jax.Array, jax.Array]
            Network outputs with shape `(..., output_channels)`. When
            `return_last_layer_inputs` is `True`, also return the activations
            passed into the output Dense layer.
        """
        if self.network_depth <= 0:
            raise ValueError("`network_depth` must be positive.")
        if self.network_width <= 0:
            raise ValueError("`network_width` must be positive.")
        if self.output_channels <= 0:
            raise ValueError("`output_channels` must be positive.")

        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.he_uniform(),
        )
        if self.use_skip_connections:
            skip_interval = max(self.network_depth // 2, 1)

        inputs = x
        for layer_index in range(self.network_depth):
            x = dense_layer(self.network_width)(x)
            x = self.activation(x)

            if (
                self.use_skip_connections
                and layer_index > 0
                and layer_index % skip_interval == 0
            ):
                x = jnp.concatenate([x, inputs], axis=-1)

        last_layer_inputs = x
        output = dense_layer(self.output_channels)(last_layer_inputs)
        if return_last_layer_inputs:
            return output, last_layer_inputs
        return output


class NeuralImage(nn.Module):
    """
    Render image intensity from normalized two-dimensional coordinates.

    Parameters:
    -----------
    positional_encoding_degree: int
        Number of positional-encoding frequency levels.
    network_depth: int
        Number of hidden MLP layers.
    network_width: int
        Width of each hidden MLP layer.
    activation: ActivationFunction
        Hidden-layer activation.
    output_channels: int
        Number of raw MLP outputs. Image rendering uses channel zero.
    use_skip_connections: bool
        Whether the MLP reinjects the encoded coordinates.
    output_logit_offset: float
        Value subtracted from the output logit before the sigmoid so
        an untrained model begins dark.
    """

    positional_encoding_degree: int = 3
    network_depth: int = 4
    network_width: int = 128
    activation: ActivationFunction = nn.relu
    output_channels: int = 1
    use_skip_connections: bool = True
    output_logit_offset: float = DEFAULT_OUTPUT_LOGIT_OFFSET

    @nn.compact
    def __call__(
        self,
        coordinates: jax.Array,
        *,
        return_last_layer_inputs: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
        """
        Evaluate image intensity at normalized coordinates.

        Parameters:
        -----------
        coordinates: jax.Array
            Coordinates with shape `(..., 2)`, conventionally in `[0, 1]`.

        Returns:
        --------
        jax.Array | tuple[jax.Array, jax.Array, jax.Array]
            Scalar image intensities with shape `coordinates.shape[:-1]`.
            When `return_last_layer_inputs` is `True`, also return the inputs
            to the output Dense layer and the channel-zero logits.
        """
        image_mlp = MLP(
            network_depth=self.network_depth,
            network_width=self.network_width,
            activation=self.activation,
            output_channels=self.output_channels,
            use_skip_connections=self.use_skip_connections,
        )

        def predict_image(
            sample_coordinates: jax.Array,
        ) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
            encoded_coordinates = positional_encode(
                sample_coordinates,
                self.positional_encoding_degree,
            )

            if return_last_layer_inputs:
                raw_output, last_layer_inputs = image_mlp(
                    encoded_coordinates,
                    return_last_layer_inputs=True,
                )
                logits = raw_output[..., 0] - self.output_logit_offset
                image = nn.sigmoid(logits)
                return image, last_layer_inputs, logits

            raw_output = image_mlp(encoded_coordinates)
            return nn.sigmoid(
                raw_output[..., 0] - self.output_logit_offset
            )

        return predict_image(coordinates)


def build_neural_image(config: NeuralImageConfig) -> NeuralImage:
    """Construct a neural image from its saved recipe."""
    return NeuralImage(
        positional_encoding_degree=config.positional_encoding_degree,
        network_depth=config.network_depth,
        network_width=config.network_width,
        output_logit_offset=config.output_logit_offset,
    )


def last_layer_inputs_and_logits(
    model: NeuralImage,
    parameters: ParameterTree,
    coordinates: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Return output-layer inputs and image logits from the reference model.


    Parameters:
    -----------
    model: NeuralImage
        Reference model whose architecture matches `parameters`.
    parameters: ParameterTree
        Flax parameter tree.
    coordinates: jax.Array
        Coordinates with shape `(number_of_pixels, 2)`.

    Returns:
    --------
    last_layer_inputs: jax.Array
        Inputs to the final dense layer.
    logits: jax.Array
        Channel-zero logits after subtracting the model output offset.
    """
    _image, last_layer_inputs, logits = model.apply(
        {"params": parameters},
        coordinates,
        return_last_layer_inputs=True,
    )
    return last_layer_inputs, logits


def last_layer_image_jacobian(
    model: NeuralImage,
    parameters: ParameterTree,
    coordinates: jax.Array,
) -> np.ndarray:
    """
    Differentiate image pixels with respect to final-layer channel-zero values.


    Parameters:
    -----------
    model: NeuralImage
        Reference coordinate model.
    parameters: ParameterTree
        Fitted Flax parameter tree.
    coordinates: jax.Array
        Flattened image coordinates.

    Returns:
    --------
    np.ndarray
        Jacobian with shape
        `(number_of_pixels, number_of_features + 1)`. The last column is the
        final-layer bias derivative.
    """
    last_layer_inputs, logits = last_layer_inputs_and_logits(
        model,
        parameters,
        coordinates,
    )
    brightness = jax.nn.sigmoid(logits)
    sigmoid_derivative = brightness * (1.0 - brightness)

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

    # (N, 1) * (N, F + 1) -> (N, F + 1).
    return np.asarray(
        sigmoid_derivative[:, None] * last_layer_inputs_with_bias
    )