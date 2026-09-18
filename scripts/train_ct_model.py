"""Train and cache a CT Fourier-feature ensemble for a scalar image."""

import argparse
import logging
from pathlib import Path

import jax

from bhuq.forward import (
    RadonProblemConfig,
    build_radon_problem_from_config,
)
from bhuq.image_loading import derive_image_name
from bhuq.model_training import train_model
from bhuq.models import (
    CoordinateConvention,
    FourierFeatureMLP,
    FourierFeatureModelConfig,
    build_coordinate_grid,
    build_fourier_feature_model,
)
from bhuq.training import TrainingConfig

LOGGER = logging.getLogger(__name__)

PIXEL_COUNT = 128
NUMBER_OF_PROJECTION_ANGLES = 40
INTERPOLATION_ORDER = 0
NOISE_STANDARD_DEVIATION = 1.0
MODEL_CONFIG = FourierFeatureModelConfig(
    number_of_frequencies=256,
    frequency_scale=4.0,
    frequency_seed=10,
    network_depth=4,
    network_width=256,
)
TRAINING_CONFIG = TrainingConfig(
    number_of_steps=5_000,
    initial_learning_rate=5e-4,
    final_learning_rate=5e-4,
    learning_rate_schedule="constant",
    batch_size=None,
    log_interval=100,
)
SEEDS = tuple(range(5))
MODEL_LABEL = "fourier_feature_mlp"
COORDINATE_CONVENTION: CoordinateConvention = "xy_exclusive"


def parse_arguments() -> argparse.Namespace:
    """Parse source, cache, and CT problem settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_image",
        type=Path,
        help="Scalar text, NumPy, or standard image file.",
    )
    parser.add_argument(
        "--source-array-key",
        help="Array key for a multi-array NPZ source.",
    )
    parser.add_argument(
        "--pixel-count",
        type=int,
        default=PIXEL_COUNT,
        help="Pixels along each square reconstruction axis.",
    )
    parser.add_argument(
        "--projection-angles",
        type=int,
        default=NUMBER_OF_PROJECTION_ANGLES,
        help="Uniform projection angles sampled over [0, pi).",
    )
    parser.add_argument(
        "--interpolation-order",
        type=int,
        choices=(0, 1),
        default=INTERPOLATION_ORDER,
        help="Rotation interpolation: 0 nearest, 1 bilinear.",
    )
    parser.add_argument(
        "--noise-standard-deviation",
        type=float,
        default=NOISE_STANDARD_DEVIATION,
        help="Standard deviation assigned to every projection sample.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("model_cache"),
        help="Parent directory for saved model bundles.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Independent ensemble seeds; defaults to five members.",
    )
    return parser.parse_args()


def build_problem_config(
    arguments: argparse.Namespace,
) -> RadonProblemConfig:
    """Build the complete problem recipe from command-line arguments."""
    return RadonProblemConfig(
        source_image_path=arguments.source_image,
        source_array_key=arguments.source_array_key,
        pixel_count=arguments.pixel_count,
        number_of_projection_angles=arguments.projection_angles,
        interpolation_order=arguments.interpolation_order,
        noise_standard_deviation=arguments.noise_standard_deviation,
    )


def build_model(
    config: FourierFeatureModelConfig,
    image_shape: tuple[int, int],
) -> tuple[FourierFeatureMLP, jax.Array]:
    """Construct the configured CT model and its coordinate grid."""
    coordinates = build_coordinate_grid(
        image_shape,
        COORDINATE_CONVENTION,
    )
    model = build_fourier_feature_model(config)
    return model, coordinates


def build_model_name(config: RadonProblemConfig) -> str:
    """Derive the model-cache grouping from source and image resolution."""
    source_name = derive_image_name(
        config.source_image_path,
        array_key=config.source_array_key,
    )
    return (
        f"ct/{source_name}/"
        f"{config.pixel_count}x{config.pixel_count}/"
        f"{MODEL_LABEL}"
    )


def main() -> None:
    """Train and cache one configured CT ensemble."""
    arguments = parse_arguments()
    problem_config = build_problem_config(arguments)
    problem, _truth = build_radon_problem_from_config(problem_config)
    model, coordinates = build_model(MODEL_CONFIG, problem.image_shape)
    model_name = build_model_name(problem_config)
    logging.basicConfig(level=logging.INFO)
    trained_model = train_model(
        model,
        coordinates,
        problem,
        TRAINING_CONFIG,
        tuple(arguments.seeds),
        coordinate_convention=COORDINATE_CONVENTION,
        model_name=model_name,
        model_metadata=MODEL_CONFIG,
        problem_metadata=problem_config,
        cache_root=arguments.cache_root,
    )
    if trained_model.saved_model is None:
        raise RuntimeError("The configured training operation was not cached.")
    LOGGER.info("Model saved to %s", trained_model.saved_model.directory)


if __name__ == "__main__":
    main()
