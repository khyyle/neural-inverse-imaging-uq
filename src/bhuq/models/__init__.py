"""Reference models and model-specific uncertainty adapters."""

from .fourier_features import (
    FourierFeatureMLP,
    build_fourier_feature_coordinate_grid,
    fourier_features,
    sample_gaussian_frequencies,
)
from .fourier_features import (
    last_layer_image_jacobian as fourier_feature_last_layer_image_jacobian,
)
from .vlbi import (
    MLP,
    NeuralImage,
    build_vlbi_coordinate_grid,
    last_layer_image_jacobian,
    last_layer_inputs_and_logits,
    positional_encode,
)

__all__ = [
    "FourierFeatureMLP",
    "MLP",
    "NeuralImage",
    "fourier_feature_last_layer_image_jacobian",
    "build_fourier_feature_coordinate_grid",
    "build_vlbi_coordinate_grid",
    "fourier_features",
    "last_layer_image_jacobian",
    "last_layer_inputs_and_logits",
    "positional_encode",
    "sample_gaussian_frequencies",
]
