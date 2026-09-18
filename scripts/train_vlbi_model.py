"""Train and cache a VLBI neural-image ensemble for an ehtim source image."""

import argparse
import logging
from pathlib import Path

import jax

from bhuq.forward import (
    EHT_2017_HIGH_BAND,
    VlbiProblemConfig,
    build_vlbi_problem_from_config,
)
from bhuq.image_loading import derive_image_name
from bhuq.model_training import train_model
from bhuq.models import (
    CoordinateConvention,
    NeuralImage,
    NeuralImageConfig,
    build_coordinate_grid,
    build_neural_image,
)
from bhuq.training import TrainingConfig

LOGGER = logging.getLogger(__name__)

PIXEL_COUNT = 100
INTEGRATION_TIME_SECONDS = 5.0
SCAN_ADVANCE_SECONDS = 600.0
START_TIME_HOURS = 0.0
STOP_TIME_HOURS = 24.0
TRANSFORM_TYPE = "nfft"
ADD_THERMAL_NOISE = False
THERMAL_NOISE_SEED = 1
MODEL_CONFIG = NeuralImageConfig(
    positional_encoding_degree=3,
    network_depth=4,
    network_width=128,
    output_logit_offset=10.0,
)
TRAINING_CONFIG = TrainingConfig()
SEEDS = tuple(range(5))
MODEL_LABEL = "neural_image"
COORDINATE_CONVENTION: CoordinateConvention = "row_column_inclusive"


def parse_arguments() -> argparse.Namespace:
    """Parse source, array, cache, and image-resolution settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_image",
        type=Path,
        help="Native ehtim text image with source metadata.",
    )
    parser.add_argument(
        "--telescope-array",
        type=Path,
        default=Path("data/vlbi/EHT2017.txt"),
        help="ehtim telescope-array file.",
    )
    parser.add_argument(
        "--pixel-count",
        type=int,
        default=PIXEL_COUNT,
        help="Pixels along each square reconstruction axis.",
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
) -> VlbiProblemConfig:
    """Build the complete problem recipe from command-line arguments."""
    return VlbiProblemConfig(
        source_image_path=arguments.source_image,
        telescope_array_path=arguments.telescope_array,
        pixel_count=arguments.pixel_count,
        bandwidth_hz=EHT_2017_HIGH_BAND.bandwidth_hz,
        integration_time_seconds=INTEGRATION_TIME_SECONDS,
        scan_advance_seconds=SCAN_ADVANCE_SECONDS,
        start_time_hours=START_TIME_HOURS,
        stop_time_hours=STOP_TIME_HOURS,
        transform_type=TRANSFORM_TYPE,
        add_thermal_noise=ADD_THERMAL_NOISE,
        thermal_noise_seed=THERMAL_NOISE_SEED,
    )


def build_model(
    config: NeuralImageConfig,
    image_shape: tuple[int, int],
) -> tuple[NeuralImage, jax.Array]:
    """Construct a neural image and its coordinate grid."""
    coordinates = build_coordinate_grid(
        image_shape,
        COORDINATE_CONVENTION,
    )
    model = build_neural_image(config)
    return model, coordinates


def build_model_name(config: VlbiProblemConfig) -> str:
    """Derive the model-cache grouping from source, array, and resolution."""
    source_name = derive_image_name(config.source_image_path)
    array_name = config.telescope_array_path.stem.lower()
    return (
        f"vlbi/{source_name}/{array_name}/"
        f"{config.pixel_count}x{config.pixel_count}/"
        f"{MODEL_LABEL}"
    )


def main() -> None:
    """Train and cache one configured VLBI ensemble."""
    arguments = parse_arguments()
    problem_config = build_problem_config(arguments)
    problem, _truth = build_vlbi_problem_from_config(problem_config)
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
