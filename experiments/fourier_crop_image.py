"""Evaluate image uncertainty for a centered Fourier-crop reconstruction."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from bhuq.evaluation import UncertaintyEvaluationConfig
from bhuq.forward import (
    FourierCropProblemConfig,
    build_fourier_crop_problem_from_config,
)
from bhuq.image_loading import derive_image_name
from bhuq.model_cache import open_saved_model
from bhuq.model_evaluation import ModelEvaluationResult, evaluate_model
from bhuq.model_training import load_model
from bhuq.models import NeuralImageConfig, build_neural_image
from bhuq.runs import ExperimentRun
from bhuq.visualization import (
    make_fourier_figure,
    make_result_figure,
    make_sparsification_figure,
    make_uncertainty_figure,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_UNCERTAINTY_CONFIG = UncertaintyEvaluationConfig(
    deformation_grid_size=64,
    deformation_prior_scale=1e-4,
    deformation_diagonal=True,
    deformation_measurement_chunk_size=8,
    deformation_map_chunk_size=8192,
    deformation_show_progress=True,
    fourier_prior_precision_fraction=1e-6,
    blind_information_fraction=1e-3,
    laplace_prior_precision=1.0,
    laplace_curvature="lanczos",
    laplace_rank=50,
    laplace_random_seed=0,
    methods=("deformation", "parameter_laplace", "ensemble"),
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration unique to this UQ result."""

    saved_model_path: Path
    uncertainty: UncertaintyEvaluationConfig = DEFAULT_UNCERTAINTY_CONFIG
    results_root: Path = Path("results")


def evaluate(config: ExperimentConfig) -> ModelEvaluationResult:
    """Reconstruct a Fourier-crop scenario and evaluate its cached ensemble."""
    saved_model = open_saved_model(config.saved_model_path)
    problem_config = FourierCropProblemConfig.from_dict(saved_model.problem_metadata)
    model_config = NeuralImageConfig.from_dict(saved_model.model_metadata)

    problem, truth = build_fourier_crop_problem_from_config(problem_config)
    model = build_neural_image(model_config)
    trained_model = load_model(saved_model, model)
    return evaluate_model(
        trained_model,
        problem,
        truth,
        config.uncertainty,
    )


def _save_figures(
    experiment: ExperimentRun,
    evaluation: ModelEvaluationResult,
) -> None:
    """Build and save the standard Fourier-crop evaluation figures."""
    result = evaluation.uncertainty
    shared_image_limits = (
        float(
            min(
                result.truth.min(),
                result.results.reconstruction.min(),
            )
        ),
        float(
            max(
                result.truth.max(),
                result.results.reconstruction.max(),
            )
        ),
    )
    figure, _ = make_result_figure(
        result,
        ("truth", "reconstruction", "pixel_error"),
        columns=3,
        value_limits={
            "truth": shared_image_limits,
            "reconstruction": shared_image_limits,
        },
    )
    experiment.save_figure("reconstruction_overview", figure)
    plt.close(figure)

    figure, _ = make_uncertainty_figure(result)
    experiment.save_figure("uncertainty_maps", figure)
    plt.close(figure)

    figure, _ = make_fourier_figure(result, columns=3)
    experiment.save_figure("fourier_maps", figure)
    plt.close(figure)

    for target in ("image", "fourier_absolute", "fourier_relative"):
        if not any(name.startswith(f"{target}/") for name in result.sparsification_results):
            continue
        figure, _ = make_sparsification_figure(result, target=target)
        experiment.save_figure(f"sparsification_{target}", figure)
        plt.close(figure)


def run(config: ExperimentConfig) -> Path:
    """Evaluate and persist one Fourier-crop uncertainty result."""
    saved_model = open_saved_model(config.saved_model_path)
    problem_config = FourierCropProblemConfig.from_dict(saved_model.problem_metadata)
    source_name = derive_image_name(
        problem_config.source_image_path,
        array_key=problem_config.source_array_key,
    )
    with ExperimentRun(
        f"fourier_crop/{source_name}",
        config,
        results_root=config.results_root,
    ) as experiment:
        evaluation = evaluate(config)
        experiment.save_evaluation("image", evaluation)
        _save_figures(experiment, evaluation)

    if experiment.run_directory is None:
        raise RuntimeError("Experiment run did not start.")
    return experiment.run_directory


def parse_arguments() -> argparse.Namespace:
    """Parse the saved model and output location."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "saved_model",
        type=Path,
        help="Saved model directory containing manifest.json.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Parent directory for experiment runs.",
    )
    return parser.parse_args()


def build_experiment_config(
    arguments: argparse.Namespace,
) -> ExperimentConfig:
    """Build the typed experiment configuration from parsed arguments."""
    return ExperimentConfig(
        saved_model_path=arguments.saved_model,
        results_root=arguments.results_root,
    )


def main() -> None:
    """Run one tracked Fourier-crop uncertainty evaluation."""
    arguments = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    run_directory = run(build_experiment_config(arguments))
    LOGGER.info("Run saved to %s", run_directory)


if __name__ == "__main__":
    main()
