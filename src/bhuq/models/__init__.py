"""Reference models and model-specific uncertainty adapters."""

from .coordinates import CoordinateConvention, build_coordinate_grid
from .fourier_features import (
    FourierFeatureMLP,
    FourierFeatureModelConfig,
    build_fourier_feature_model,
    fourier_features,
    sample_gaussian_frequencies,
)
from .fourier_features import (
    last_layer_image_jacobian as fourier_feature_last_layer_image_jacobian,
)
from .vlbi import (
    MLP,
    NeuralImage,
    NeuralImageConfig,
    build_neural_image,
    last_layer_image_jacobian,
    last_layer_inputs_and_logits,
    positional_encode,
)

__all__ = [
    "CoordinateConvention",
    "FourierFeatureModelConfig",
    "FourierFeatureMLP",
    "MLP",
    "NeuralImage",
    "NeuralImageConfig",
    "fourier_feature_last_layer_image_jacobian",
    "build_coordinate_grid",
    "build_fourier_feature_model",
    "build_neural_image",
    "fourier_features",
    "last_layer_image_jacobian",
    "last_layer_inputs_and_logits",
    "positional_encode",
    "sample_gaussian_frequencies",
]
