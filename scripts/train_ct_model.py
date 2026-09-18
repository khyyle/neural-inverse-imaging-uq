"""Train and cache a CT Sgr A* Fourier-feature ensemble."""

import logging
from pathlib import Path

import ehtim as eh
import jax
import numpy as np
from skimage.transform import resize_local_mean

from bhuq.forward import RadonProblemConfig, build_radon_inverse_problem
from bhuq.model_training import train_model
from bhuq.models import (
    FourierFeatureMLP,
    FourierFeatureModelConfig,
    build_fourier_feature_coordinate_grid,
    sample_gaussian_frequencies,
)
from bhuq.training import TrainingConfig

LOGGER = logging.getLogger(__name__)


def _default_training_config() -> TrainingConfig:
    return TrainingConfig(
        number_of_steps=5_000,
        initial_learning_rate=5e-4,
        final_learning_rate=5e-4,
        learning_rate_schedule="constant",
        batch_size=None,
        log_interval=100,
    )


PROBLEM = RadonProblemConfig(
    source_image_path=Path("data/images/avery_sgra_eofn.txt"),
    pixel_count=128,
    number_of_projection_angles=40,
    interpolation_order=0,
    noise_standard_deviation=1.0,
)
MODEL = FourierFeatureModelConfig(
    number_of_frequencies=256,
    frequency_scale=4.0,
    frequency_seed=10,
    network_depth=4,
    network_width=256,
)
TRAINING = _default_training_config()
SEEDS = tuple(range(5))
MODEL_NAME = "ct/sgr-a/128x128/fourier_feature_mlp"
CACHE_ROOT = Path("model_cache")


def main() -> None:

    # set up ground truth image
    source = eh.image.load_txt(str(PROBLEM.source_image_path))
    source_image = np.asarray(source.imarr(), dtype=np.float32)
    resized_image = resize_local_mean(
        source_image,
        (PROBLEM.pixel_count, PROBLEM.pixel_count),
        grid_mode=True,
        preserve_range=True,
    )
    truth = resized_image / float(resized_image.max()) # normalize pixel values to 0-1

    # build the inverse problem
    angles = np.linspace(
        0.0,
        np.pi,
        PROBLEM.number_of_projection_angles,
        endpoint=False,
        dtype=np.float32,
    )
    problem = build_radon_inverse_problem(
        truth,
        angles,
        noise_standard_deviation=(
            PROBLEM.noise_standard_deviation
        ),
        interpolation_order=PROBLEM.interpolation_order,
    )

    # setup a coordiante MLP with gaussian sampled fourier features
    coordinates = build_fourier_feature_coordinate_grid(problem.image_shape)
    frequencies = sample_gaussian_frequencies(
        jax.random.PRNGKey(MODEL.frequency_seed),
        number_of_frequencies=MODEL.number_of_frequencies,
        scale=MODEL.frequency_scale,
    )
    model = FourierFeatureMLP(
        frequency_matrix=frequencies,
        network_depth=MODEL.network_depth,
        network_width=MODEL.network_width,
    )

    logging.basicConfig(level=logging.INFO)
    trained_model = train_model(
        model,
        coordinates,
        problem,
        TRAINING,
        SEEDS,
        model_name=MODEL_NAME,
        model_metadata=MODEL,
        problem_metadata=PROBLEM,
        input_paths={"source_image": PROBLEM.source_image_path},
        cache_root=CACHE_ROOT,
    )
    if trained_model.saved_model is None:
        raise RuntimeError("The configured training operation was not cached.")
    LOGGER.info(
        "Model saved to %s",
        trained_model.saved_model.directory,
    )


if __name__ == "__main__":
    main()
