"""VLBI observation simulation and linear visibility-operator construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .linear_problem import LinearInverseProblem
from .operator import DenseLinearOperator


@dataclass(frozen=True)
class VlbiBand:
    reference_frequency_hz: float
    bandwidth_hz: float

    def __post_init__(self) -> None:
        """Require positive frequency and bandwidth values."""
        if self.reference_frequency_hz <= 0.0:
            raise ValueError("`reference_frequency_hz` must be positive.")
        if self.bandwidth_hz <= 0.0:
            raise ValueError("`bandwidth_hz` must be positive.")


# EHT-2017 observed two independent 2 GHz bands:
# https://doi.org/10.3847/2041-8213/ab0c57
EHT_2017_LOW_BAND = VlbiBand(
    reference_frequency_hz=227_100_000_000.0,
    bandwidth_hz=2_000_000_000.0,
)
EHT_2017_HIGH_BAND = VlbiBand(
    reference_frequency_hz=229_100_000_000.0,
    bandwidth_hz=2_000_000_000.0,
)


def simulate_observation(
    source_image: Any,
    telescope_array: Any,
    *,
    bandwidth_hz: float,
    integration_time_seconds: float = 5.0,
    scan_advance_seconds: float = 600.0,
    start_time_hours: float = 0.0,
    stop_time_hours: float = 24.0,
    transform_type: str = "nfft",
    add_thermal_noise: bool = False,
) -> Any:
    """
    Sample an ehtim image with an ehtim telescope array.

    Parameters:
    -----------
    source_image: ehtim.image.Image
        Monochromatic sky-brightness image. Its reference frequency is used
        as the observation frequency.
    telescope_array: ehtim.array.Array
        Telescope coordinates and sensitivity values used to generate
        baselines and thermal-noise standard deviations.
    bandwidth_hz: float
        Effective bandwidth of the observation.
    integration_time_seconds: float
        Integration time for each visibility measurement.
    scan_advance_seconds: float
        Time between successive scans.
    start_time_hours: float
        Beginning of the observation in UTC hours.
    stop_time_hours: float
        End of the observation in UTC hours.
    transform_type: str
        Fourier-transform implementation passed to ehtim, one of ['fast', 'nfft', 'direct']
    add_thermal_noise: bool
        If `True`, draw thermal noise and calibration corruptions through
        `observe_same`. If `False`, preserve noiseless visibilities while
        retaining the array-derived noise standard deviations.

    Returns:
    --------
    ehtim.obsdata.Obsdata
        Observation containing sampled complex visibilities, baseline
        coordinates, and per-visibility noise values.

    Raises:
    -------
    ValueError
        If `bandwidth_hz` is not positive.
    """
    if bandwidth_hz <= 0.0:
        raise ValueError("`bandwidth_hz` must be positive.")

    observation_schedule = telescope_array.obsdata(
        tint=integration_time_seconds,
        tadv=scan_advance_seconds,
        tstart=start_time_hours,
        tstop=stop_time_hours,
        ra=source_image.ra,
        dec=source_image.dec,
        rf=source_image.rf,
        mjd=source_image.mjd,
        bw=bandwidth_hz,
        timetype="UTC",
        polrep="stokes",
    )

    if add_thermal_noise:
        return source_image.observe_same(observation_schedule, ttype=transform_type)
    return source_image.observe_same_nonoise(observation_schedule, ttype=transform_type)


def build_vlbi_inverse_problem(
    observation: Any,
    *,
    pixel_count: int,
    field_of_view_radians: float,
) -> LinearInverseProblem:
    """
    Extract the linear visibility equation from an ehtim observation.

    `ehtim.imaging.imager_utils.chisqdata_vis` returns the observed
    visibilities, their noise standard deviations, and the matrix `A` for the
    equation `visibilities = A @ image`. This ehtim function
    always constructs a dense direct Fourier matrix; the transform selected
    while simulating the observation does not change that matrix construction.

    Parameters:
    -----------
    observation: ehtim.obsdata.Obsdata
        Observation produced by `simulate_observation` or loaded elsewhere.
    pixel_count: int
        Number of pixels along each side of the square reconstruction grid.
    field_of_view_radians: float
        Angular width of the square reconstruction grid in radians.
    Returns:
    --------
    LinearInverseProblem
        Visibility matrix, measured visibilities, thermal-noise standard
        deviations, and image dimensions.

    Raises:
    -------
    ValueError
        If `pixel_count` or `field_of_view_radians` is not positive.
    """
    if pixel_count <= 0:
        raise ValueError("`pixel_count` must be positive.")
    if field_of_view_radians <= 0.0:
        raise ValueError("`field_of_view_radians` must be positive.")

    import ehtim as eh

    prior_image = eh.image.make_square(observation, pixel_count, field_of_view_radians)
    visibilities, noise_standard_deviation, A = (
        eh.imaging.imager_utils.chisqdata_vis(
            observation,
            prior_image,
            mask=[],
        )
    )
    return LinearInverseProblem(
        operator=DenseLinearOperator(matrix=np.asarray(A), image_shape=(pixel_count, pixel_count)),
        observed_measurements=np.asarray(visibilities),
        noise_standard_deviation=np.asarray(noise_standard_deviation),
        field_of_view_microarcseconds=(
            field_of_view_radians / eh.RADPERUAS
        ),
        description=(
            f"VLBI visibility operator with {np.asarray(visibilities).size} measurements"
        ),
    )
