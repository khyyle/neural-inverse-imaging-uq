"""Evaluate CT uncertainty for a caller-selected model bundle."""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import ehtim as eh
import jax
import numpy as np
from skimage.transform import resize_local_mean

from bhuq.evaluation import UncertaintyEvaluationConfig
from bhuq.forward import build_radon_inverse_problem
from bhuq.model_cache import open_saved_model
from bhuq.model_evaluation import (
    ModelEvaluationResult,
    evaluate_model,
)
from bhuq.model_training import load_model
from bhuq.models import (
    FourierFeatureMLP,
    build_fourier_feature_coordinate_grid,
    sample_gaussian_frequencies,
)
from bhuq.runs import ExperimentRun

LOGGER = logging.getLogger(__name__)


def _default_uncertainty_config() -> UncertaintyEvaluationConfig:
    return UncertaintyEvaluationConfig(
        deformation_grid_size=64,
        deformation_prior_scale=1e-8,
        methods=("deformation", "parameter_laplace", "ensemble"),
    )


@dataclass(frozen=True)
class ExperimentConfig:
    saved_model_path: Path
    uncertainty: UncertaintyEvaluationConfig = field(
        default_factory=_default_uncertainty_config
    )
    results_root: Path = Path("results")


def evaluate(config: ExperimentConfig) -> ModelEvaluationResult:
    """Reconstruct the CT scenario and evaluate its cached ensemble."""
    saved_model = open_saved_model(config.saved_model_path)
    problem_metadata = saved_model.problem_metadata
    model_metadata = saved_model.model_metadata

    source_image_path = Path(problem_metadata["source_image_path"])
    source = eh.image.load_txt(str(source_image_path))
    source_image = np.asarray(source.imarr(), dtype=np.float32)
    pixel_count = problem_metadata["pixel_count"]
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
        problem_metadata["number_of_projection_angles"],
        endpoint=False,
        dtype=np.float32,
    )
    problem = build_radon_inverse_problem(
        truth,
        angles,
        noise_standard_deviation=(
            problem_metadata["noise_standard_deviation"]
        ),
        interpolation_order=problem_metadata["interpolation_order"],
    )
    coordinates = build_fourier_feature_coordinate_grid(problem.image_shape)
    frequencies = sample_gaussian_frequencies(
        jax.random.PRNGKey(model_metadata["frequency_seed"]),
        number_of_frequencies=model_metadata["number_of_frequencies"],
        scale=model_metadata["frequency_scale"],
    )
    model = FourierFeatureMLP(
        frequency_matrix=frequencies,
        network_depth=model_metadata["network_depth"],
        network_width=model_metadata["network_width"],
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
