"""Evaluate CT uncertainty for a caller-selected model bundle."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import ehtim as eh
import jax
import numpy as np
from skimage.transform import resize_local_mean

from bhuq.evaluation import UncertaintyEvaluationConfig
from bhuq.forward import RadonProblemConfig, build_radon_inverse_problem
from bhuq.model_cache import open_saved_model
from bhuq.model_evaluation import (
    ModelEvaluationResult,
    evaluate_model,
)
from bhuq.model_training import load_model
from bhuq.models import (
    FourierFeatureMLP,
    FourierFeatureModelConfig,
    build_fourier_feature_coordinate_grid,
    sample_gaussian_frequencies,
)
from bhuq.runs import ExperimentRun

LOGGER = logging.getLogger(__name__)

DEFAULT_UNCERTAINTY_CONFIG = UncertaintyEvaluationConfig(
    deformation_grid_size=64,
    deformation_prior_scale=1e-8,
    methods=("deformation", "parameter_laplace", "ensemble"),
)


@dataclass(frozen=True)
class ExperimentConfig:
    saved_model_path: Path
    uncertainty: UncertaintyEvaluationConfig = DEFAULT_UNCERTAINTY_CONFIG
    results_root: Path = Path("results")


def evaluate(config: ExperimentConfig) -> ModelEvaluationResult:
    """Reconstruct a CT scenario and evaluate its cached ensemble."""
    saved_model = open_saved_model(config.saved_model_path)
    problem_config = RadonProblemConfig.from_dict(
        saved_model.problem_metadata
    )
    model_config = FourierFeatureModelConfig.from_dict(
        saved_model.model_metadata
    )

    source_image_path = problem_config.source_image_path
    source = eh.image.load_txt(str(source_image_path))
    source_image = np.asarray(source.imarr(), dtype=np.float32)
    pixel_count = problem_config.pixel_count
    resized_image = resize_local_mean(
        source_image,
        (pixel_count, pixel_count),
        grid_mode=True,
        preserve_range=True,
    )
    truth = resized_image / float(resized_image.max())
    angles = np.linspace(
        0.0,
        np.pi,
        problem_config.number_of_projection_angles,
        endpoint=False,
        dtype=np.float32,
    )
    problem = build_radon_inverse_problem(
        truth,
        angles,
        noise_standard_deviation=problem_config.noise_standard_deviation,
        interpolation_order=problem_config.interpolation_order,
    )
    coordinates = build_fourier_feature_coordinate_grid(problem.image_shape)
    frequencies = sample_gaussian_frequencies(
        jax.random.PRNGKey(model_config.frequency_seed),
        number_of_frequencies=model_config.number_of_frequencies,
        scale=model_config.frequency_scale,
    )
    model = FourierFeatureMLP(
        frequency_matrix=frequencies,
        network_depth=model_config.network_depth,
        network_width=model_config.network_width,
    )

    input_paths = {"source_image": source_image_path}
    trained_model = load_model(
        saved_model,
        model,
        coordinates,
        input_paths=input_paths,
    )
    return evaluate_model(
        trained_model,
        model,
        coordinates,
        problem,
        truth,
        config.uncertainty,
    )


def run(config: ExperimentConfig) -> Path:
    """Evaluate and optionally persist this coherent research result."""
    with ExperimentRun(
        "ct_black_hole",
        config,
        results_root=config.results_root,
    ) as experiment:
        evaluation = evaluate(config)
        experiment.save_evaluation("sgr_a", evaluation)
    if experiment.run_directory is None:
        raise RuntimeError("Experiment run did not start.")
    return experiment.run_directory


def main() -> None:
    """Read the model bundle path and run the tracked experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_model", type=Path)
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_directory = run(
        ExperimentConfig(saved_model_path=arguments.saved_model)
    )
    LOGGER.info("Run saved to %s", run_directory)


if __name__ == "__main__":
    main()
