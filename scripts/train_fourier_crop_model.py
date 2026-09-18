"""Train and cache a neural-image ensemble from centered Fourier measurements."""

import argparse
import logging
from pathlib import Path

import jax

from bhuq.forward import (
    FourierCropProblemConfig,
    LinearInverseProblem,
    build_fourier_crop_problem_from_config,
)
from bhuq.image_loading import derive_image_name
from bhuq.model_training import TrainedModel, train_model
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
CROP_SIZE = 50
NOISE_STANDARD_DEVIATION = 1e-4
MODEL_CONFIG = NeuralImageConfig(
    positional_encoding_degree=3,
    network_depth=4,
    network_width=128,
    output_logit_offset=0.0,
)
TRAINING_CONFIG = TrainingConfig(
    number_of_steps=10_000,
    initial_learning_rate=1e-3,
    final_learning_rate=1e-5,
    learning_rate_schedule="cosine",
    batch_size=None,
    log_interval=100,
)
SEEDS = tuple(range(5))
MODEL_LABEL = "neural_image"
COORDINATE_CONVENTION: CoordinateConvention = "row_column_inclusive"


def parse_arguments() -> argparse.Namespace:
    """Parse source, Fourier problem, cache, and executor settings."""
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
        "--crop-size",
        type=int,
        default=CROP_SIZE,
        help="Width and height of the centered Fourier crop.",
    )
    parser.add_argument(
        "--noise-standard-deviation",
        type=float,
        default=NOISE_STANDARD_DEVIATION,
        help="Per-component coefficient noise after image normalization.",
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
    parser.add_argument(
        "--executor",
        choices=("local", "modal"),
        default="local",
        help="Training executor; Modal assigns one GPU to each seed.",
    )
    parser.add_argument(
        "--modal-gpu",
        help="Modal GPU resource assigned to each member; defaults to A10G.",
    )
    parser.add_argument(
        "--modal-max-parallel",
        type=int,
        help="Maximum Modal members running concurrently.",
    )
    return parser.parse_args()


def build_problem_config(
    arguments: argparse.Namespace,
) -> FourierCropProblemConfig:
    """Build the complete problem recipe from command-line arguments."""
    return FourierCropProblemConfig(
        source_image_path=arguments.source_image,
        source_array_key=arguments.source_array_key,
        pixel_count=arguments.pixel_count,
        crop_size=arguments.crop_size,
        noise_standard_deviation=arguments.noise_standard_deviation,
    )


def build_model(
    config: NeuralImageConfig,
    image_shape: tuple[int, int],
) -> tuple[NeuralImage, jax.Array]:
    """Construct the configured neural image and coordinate grid."""
    coordinates = build_coordinate_grid(
        image_shape,
        COORDINATE_CONVENTION,
    )
    model = build_neural_image(config)
    return model, coordinates


def build_training_objects(
    problem_config: FourierCropProblemConfig,
    model_config: NeuralImageConfig,
) -> tuple[NeuralImage, jax.Array, LinearInverseProblem]:
    """Construct the complete runtime state for local or remote training."""
    problem, _truth = build_fourier_crop_problem_from_config(problem_config)
    model, coordinates = build_model(model_config, problem.image_shape)
    return model, coordinates, problem


def build_model_name(config: FourierCropProblemConfig) -> str:
    """Derive model-cache grouping from source, image, and crop dimensions."""
    source_name = derive_image_name(
        config.source_image_path,
        array_key=config.source_array_key,
    )
    return (
        f"fourier_crop/{source_name}/"
        f"{config.pixel_count}x{config.pixel_count}/"
        f"{config.crop_size}x{config.crop_size}/"
        f"{MODEL_LABEL}"
    )


def train(
    arguments: argparse.Namespace,
    problem_config: FourierCropProblemConfig,
    model_name: str,
) -> TrainedModel:
    """Train with the selected executor and return one standard model."""
    seeds = tuple(arguments.seeds)
    if arguments.executor == "local":
        if arguments.modal_gpu is not None or arguments.modal_max_parallel is not None:
            raise ValueError("Modal resource options require `--executor modal`.")
        model, coordinates, problem = build_training_objects(
            problem_config,
            MODEL_CONFIG,
        )
        return train_model(
            model,
            coordinates,
            problem,
            TRAINING_CONFIG,
            seeds,
            coordinate_convention=COORDINATE_CONVENTION,
            model_name=model_name,
            model_metadata=MODEL_CONFIG,
            problem_metadata=problem_config,
            cache_root=arguments.cache_root,
        )

    try:
        from bhuq.modal_training import (
            ModalExecutionConfig,
            ModalTrainingRequest,
            train_ensemble_on_modal,
        )
    except ModuleNotFoundError as error:
        if error.name != "modal":
            raise
        raise SystemExit("Modal training requires `uv sync --extra modal`.") from error

    execution_values = {
        "max_parallel": arguments.modal_max_parallel,
    }
    if arguments.modal_gpu is not None:
        execution_values["gpu"] = arguments.modal_gpu
    request = ModalTrainingRequest.from_configs(
        builder_path="train_fourier_crop_model:build_training_objects",
        problem_config=problem_config,
        model_config=MODEL_CONFIG,
        training_config=TRAINING_CONFIG,
        coordinate_convention=COORDINATE_CONVENTION,
        model_name=model_name,
    )
    return train_ensemble_on_modal(
        request,
        seeds,
        execution=ModalExecutionConfig(**execution_values),
        cache_root=arguments.cache_root,
    )


def main() -> None:
    """Train and cache one configured Fourier-crop ensemble."""
    arguments = parse_arguments()
    problem_config = build_problem_config(arguments)
    model_name = build_model_name(problem_config)
    logging.basicConfig(level=logging.INFO)
    trained_model = train(arguments, problem_config, model_name)
    if trained_model.saved_model is None:
        raise RuntimeError("The configured training operation was not cached.")
    LOGGER.info("Model saved to %s", trained_model.saved_model.directory)


if __name__ == "__main__":
    main()
