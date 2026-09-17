"""Train and cache a VLBI Sgr A* neural-image ensemble."""

import logging
from dataclasses import dataclass
from pathlib import Path

import ehtim as eh

from bhuq.forward import (
    EHT_2017_HIGH_BAND,
    build_vlbi_inverse_problem,
    simulate_observation,
)
from bhuq.model_training import train_model
from bhuq.models import NeuralImage, build_vlbi_coordinate_grid
from bhuq.training import TrainingConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProblemConfig:
    source_image_path: Path = Path("data/images/avery_sgra_eofn.txt")
    telescope_array_path: Path = Path("data/vlbi/EHT2017.txt")
    pixel_count: int = 100
    bandwidth_hz: float = EHT_2017_HIGH_BAND.bandwidth_hz
    integration_time_seconds: float = 5.0
    scan_advance_seconds: float = 600.0
    start_time_hours: float = 0.0
    stop_time_hours: float = 24.0
    transform_type: str = "nfft"
    add_thermal_noise: bool = False


@dataclass(frozen=True)
class ModelConfig:
    positional_encoding_degree: int = 3
    network_depth: int = 4
    network_width: int = 128
    output_logit_offset: float = 10.0


PROBLEM = ProblemConfig()
MODEL = ModelConfig()
TRAINING = TrainingConfig()
SEEDS = tuple(range(5))
MODEL_NAME = "vlbi/sgr-a/eht2017/100x100/neural_image"
CACHE_ROOT = Path("model_cache")


def main() -> None:
    # set up observation and ground truth
    source = eh.image.load_txt(str(PROBLEM.source_image_path))
    telescope_array = eh.array.load_txt(str(PROBLEM.telescope_array_path))
    observation = simulate_observation(
        source,
        telescope_array,
        bandwidth_hz=PROBLEM.bandwidth_hz,
        integration_time_seconds=PROBLEM.integration_time_seconds,
        scan_advance_seconds=PROBLEM.scan_advance_seconds,
        start_time_hours=PROBLEM.start_time_hours,
        stop_time_hours=PROBLEM.stop_time_hours,
        transform_type=PROBLEM.transform_type,
        add_thermal_noise=PROBLEM.add_thermal_noise,
    )

    # build the inverse problem
    problem = build_vlbi_inverse_problem(
        observation,
        pixel_count=PROBLEM.pixel_count,
        field_of_view_radians=source.fovx(),
    )
    coordinates = build_vlbi_coordinate_grid(problem.image_shape)

    # setup coordinate MLP
    model = NeuralImage(
        positional_encoding_degree=(
            MODEL.positional_encoding_degree
        ),
        network_depth=MODEL.network_depth,
        network_width=MODEL.network_width,
        output_logit_offset=MODEL.output_logit_offset,
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
        input_paths={
            "source_image": PROBLEM.source_image_path,
            "telescope_array": PROBLEM.telescope_array_path,
        },
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
