"""Evaluate VLBI uncertainty for a caller-selected model bundle."""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import ehtim as eh
import numpy as np

from bhuq.evaluation import UncertaintyEvaluationConfig
from bhuq.forward import (
    build_vlbi_inverse_problem,
    simulate_observation,
)
from bhuq.model_cache import open_saved_model
from bhuq.model_evaluation import (
    ModelEvaluationResult,
    evaluate_model,
)
from bhuq.model_training import load_model
from bhuq.models import (
    NeuralImage,
    build_vlbi_coordinate_grid,
)
from bhuq.runs import ExperimentRun

LOGGER = logging.getLogger(__name__)


def _default_uncertainty_config() -> UncertaintyEvaluationConfig:
    """Return the UQ methods compared by this experiment."""
    return UncertaintyEvaluationConfig(
        deformation_grid_size=64,
        deformation_prior_scale=1e-4,
        methods=(
            "deformation",
            "parameter_laplace",
            "ensemble",
            "fourier_data_blind",
        ),
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration unique to this UQ result."""

    saved_model_path: Path
    uncertainty: UncertaintyEvaluationConfig = field(
        default_factory=_default_uncertainty_config
    )
    results_root: Path = Path("results")


def evaluate(config: ExperimentConfig) -> ModelEvaluationResult:
    """Reconstruct the VLBI scenario and evaluate its cached ensemble."""
    saved_model = open_saved_model(config.saved_model_path)
    problem_metadata = saved_model.problem_metadata
    model_metadata = saved_model.model_metadata

    source_image_path = Path(problem_metadata["source_image_path"])
    telescope_array_path = Path(problem_metadata["telescope_array_path"])
    source = eh.image.load_txt(str(source_image_path))
    telescope_array = eh.array.load_txt(str(telescope_array_path))
    observation = simulate_observation(
        source,
        telescope_array,
        bandwidth_hz=problem_metadata["bandwidth_hz"],
        integration_time_seconds=(
            problem_metadata["integration_time_seconds"]
        ),
        scan_advance_seconds=problem_metadata["scan_advance_seconds"],
        start_time_hours=problem_metadata["start_time_hours"],
        stop_time_hours=problem_metadata["stop_time_hours"],
        transform_type=problem_metadata["transform_type"],
        add_thermal_noise=problem_metadata["add_thermal_noise"],
    )
    problem = build_vlbi_inverse_problem(
        observation,
        pixel_count=problem_metadata["pixel_count"],
        field_of_view_radians=source.fovx(),
    )
    coordinates = build_vlbi_coordinate_grid(problem.image_shape)
    model = NeuralImage(**model_metadata)
    truth = np.asarray(
        source.regrid_image(
            source.fovx(),
            problem_metadata["pixel_count"],
        ).imarr()
    )

    input_paths = {
        "source_image": source_image_path,
        "telescope_array": telescope_array_path,
    }
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
        "vlbi_black_hole",
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
