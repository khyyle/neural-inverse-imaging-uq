"""Train and cache a VLBI Sgr A* neural-image ensemble."""

import logging
from pathlib import Path

import ehtim as eh

from bhuq.forward import (
    EHT_2017_HIGH_BAND,
    VlbiProblemConfig,
    build_vlbi_inverse_problem,
    simulate_observation,
)
from bhuq.model_training import train_model
from bhuq.models import (
    NeuralImage,
    NeuralImageConfig,
    build_vlbi_coordinate_grid,
)
from bhuq.training import TrainingConfig

LOGGER = logging.getLogger(__name__)


PROBLEM = VlbiProblemConfig(
    source_image_path=Path("data/images/avery_sgra_eofn.txt"),
    telescope_array_path=Path("data/vlbi/EHT2017.txt"),
    pixel_count=100,
    bandwidth_hz=EHT_2017_HIGH_BAND.bandwidth_hz,
    integration_time_seconds=5.0,
    scan_advance_seconds=600.0,
    start_time_hours=0.0,
    stop_time_hours=24.0,
    transform_type="nfft",
    add_thermal_noise=False,
    thermal_noise_seed=1,
)
MODEL = NeuralImageConfig(
    positional_encoding_degree=3,
    network_depth=4,
    network_width=128,
    output_logit_offset=10.0,
)
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
        thermal_noise_seed=PROBLEM.thermal_noise_seed,
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
