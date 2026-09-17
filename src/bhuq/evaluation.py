"""uncertainty evaluation for fitted two-dimensional image models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import numpy as np

from .forward import LinearInverseProblem
from .metrics import (
    area_under_sparsification_error,
    fourier_error_maps,
    masked_fourier_content,
    masked_fourier_content_correlation,
    masked_fourier_power_fraction,
    spearman_rank_correlation,
)
from .uq.deformation import deformation_laplace
from .uq.ensemble import ensemble_uncertainty
from .uq.fourier import fourier_data_blind_map
from .uq.laplace import (
    LaplaceCurvature,
    linearized_parameter_laplace,
)

type UncertaintyMethod = Literal[
    "deformation",
    "parameter_laplace",
    "ensemble",
    "fourier_data_blind",
]

SUPPORTED_UNCERTAINTY_METHODS: tuple[UncertaintyMethod, ...] = (
    "deformation",
    "parameter_laplace",
    "ensemble",
    "fourier_data_blind",
)


@dataclass(frozen=True)
class UncertaintyEvaluationConfig:
    """
    Configure shared uncertainty calculations.

    Parameters:
    -----------
    deformation_grid_size: int
        Nodes per side of the deformation grid.
    deformation_prior_scale: float
        Prior precision numerator. Evaluation divides it by the number of
        deformation nodes.
    deformation_diagonal: bool
        Whether to use the BayesRays diagonal Fisher approximation.
    deformation_measurement_chunk_size: int
        Measurement gradients materialized together.
    deformation_map_chunk_size: int
        Image coordinates used together for covariance propagation.
    deformation_show_progress: bool
        Whether to display deformation Fisher chunk progress.
    fourier_prior_precision_fraction: float
        Fourier prior precision relative to maximum data information.
    blind_information_fraction: float
        Information threshold defining data-blind Fourier modes.
    laplace_prior_precision: float
        Isotropic Gaussian parameter-prior precision.
    laplace_curvature: LaplaceCurvature
        Full dense or low-rank Lanczos GGN curvature.
    laplace_rank: int
        Number of Lanczos eigenpairs retained for full-network Laplace.
    laplace_random_seed: int
        Seed used to initialize Lanczos.
    methods: tuple[UncertaintyMethod, ...]
        UQ methods evaluated together. Individual method functions remain
        independently callable.
    """

    deformation_grid_size: int
    deformation_prior_scale: float
    deformation_diagonal: bool = True
    deformation_measurement_chunk_size: int = 8
    deformation_map_chunk_size: int = 8192
    deformation_show_progress: bool = True
    fourier_prior_precision_fraction: float = 1e-6
    blind_information_fraction: float = 1e-3
    laplace_prior_precision: float = 1.0
    laplace_curvature: LaplaceCurvature = "lanczos"
    laplace_rank: int = 50
    laplace_random_seed: int = 0
    methods: tuple[UncertaintyMethod, ...] = SUPPORTED_UNCERTAINTY_METHODS

    def __post_init__(self) -> None:
        """Reject unknown or repeated UQ methods."""
        unknown_methods = set(self.methods) - set(SUPPORTED_UNCERTAINTY_METHODS)
        if unknown_methods:
            raise ValueError(
                f"Unsupported uncertainty methods: {sorted(unknown_methods)}."
            )
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("`methods` must not contain duplicates.")
        if self.laplace_prior_precision <= 0.0:
            raise ValueError("`laplace_prior_precision` must be positive.")
        if self.laplace_curvature not in ("full", "lanczos"):
            raise ValueError(
                "`laplace_curvature` must be `full` or `lanczos`."
            )
        if self.laplace_rank <= 0:
            raise ValueError("`laplace_rank` must be positive.")


@dataclass(frozen=True)
class UncertaintyMaps:
    """
    Parameters:
    -----------
    number_of_measurements: int
        Number of measurements used to calculate uncertainty.
    reconstruction: np.ndarray
        Selected reconstruction.
    member_images: np.ndarray
        Every rendered model member with shape `(members, height, width)`.
    deformation_fisher_information: np.ndarray | None
        Deformation Fisher diagonal or full matrix when selected.
    deformation_posterior_covariance: np.ndarray | None
        Deformation posterior variance vector or full covariance matrix.
    deformation_node_variance: np.ndarray | None
        Marginal variance at every two-channel deformation node.
    deformation_uncertainty: np.ndarray | None
        Spatial deformation standard-deviation map when selected.
    parameter_laplace_standard_deviation: np.ndarray | None
        Image standard deviation from selected parameter uncertainty.
    ensemble_mean_image: np.ndarray | None
        Mean reconstruction across ensemble members when selected.
    ensemble_standard_deviation: np.ndarray | None
        Pixel standard deviation across ensemble members when selected.
    ensemble_fourier_variance: np.ndarray | None
        Fourier variance across ensemble members when selected.
    fourier_information: np.ndarray | None
        Data information per Fourier mode when requested.
    fourier_data_blind_variance: np.ndarray | None
        Posterior variance of additive Fourier modes when requested.
    fourier_blind_mask: np.ndarray | None
        Boolean data-blind Fourier mask when requested.
    blind_content: np.ndarray | None
        Reconstruction content occupying data-blind modes.
    """

    number_of_measurements: int
    reconstruction: np.ndarray
    member_images: np.ndarray
    deformation_fisher_information: np.ndarray | None
    deformation_posterior_covariance: np.ndarray | None
    deformation_node_variance: np.ndarray | None
    deformation_uncertainty: np.ndarray | None
    parameter_laplace_standard_deviation: np.ndarray | None
    ensemble_mean_image: np.ndarray | None
    ensemble_standard_deviation: np.ndarray | None
    ensemble_fourier_variance: np.ndarray | None
    fourier_information: np.ndarray | None
    fourier_data_blind_variance: np.ndarray | None
    fourier_blind_mask: np.ndarray | None
    blind_content: np.ndarray | None

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Return compact derived products using stable artifact names."""
        arrays = {"reconstruction": self.reconstruction}
        optional_arrays = {
            "deformation_fisher_information": self.deformation_fisher_information,
            "deformation_posterior_covariance": self.deformation_posterior_covariance,
            "deformation_node_variance": self.deformation_node_variance,
            "deformation_uncertainty": self.deformation_uncertainty,
            "parameter_laplace_pixel_standard_deviation": self.parameter_laplace_standard_deviation,
            "ensemble_mean_image": self.ensemble_mean_image,
            "ensemble_pixel_standard_deviation": self.ensemble_standard_deviation,
            "ensemble_fourier_variance": self.ensemble_fourier_variance,
            "fourier_information": self.fourier_information,
            "fourier_data_blind_variance": self.fourier_data_blind_variance,
            "fourier_blind_mask": self.fourier_blind_mask,
            "blind_content": self.blind_content,
        }
        arrays.update(
            {
                name: array
                for name, array in optional_arrays.items()
                if array is not None
            }
        )
        return arrays


@dataclass(frozen=True)
class UncertaintyEvaluationResult:
    """
    Parameters:
    -----------
    maps: UncertaintyMaps
        Truth-independent uncertainty calculation.
    truth: np.ndarray
        Ground-truth image.
    pixel_error: np.ndarray
        Absolute image-space reconstruction error.
    absolute_fourier_error: np.ndarray
        Absolute Fourier-domain reconstruction error.
    relative_fourier_error: np.ndarray
        Relative Fourier-domain reconstruction error.
    metrics: dict[str, Any]
        Scalar and structured validation metrics.
    """

    maps: UncertaintyMaps
    truth: np.ndarray
    pixel_error: np.ndarray
    absolute_fourier_error: np.ndarray
    relative_fourier_error: np.ndarray
    metrics: dict[str, Any]

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Return uncertainty maps together with truth-based error arrays."""
        return {
            **self.maps.arrays,
            "truth": self.truth,
            "pixel_error": self.pixel_error,
            "absolute_fourier_error": self.absolute_fourier_error,
            "relative_fourier_error": self.relative_fourier_error,
        }


def compute_uncertainty_maps(
    *,
    forward_model: LinearInverseProblem,
    reconstruction: np.ndarray,
    reference_parameters: Any,
    member_images: np.ndarray,
    coordinates: jax.Array,
    render_at_coordinates: Callable[[Any, jax.Array], jax.Array],
    config: UncertaintyEvaluationConfig,
) -> UncertaintyMaps:
    """
    Compute selected uncertainty maps without requiring ground truth.

    Parameters:
    -----------
    forward_model: LinearInverseProblem
        Measurement operator and noise model.
    reconstruction: np.ndarray
        Reconstruction selected by the caller.
    reference_parameters: Any
        Parameters corresponding to `reconstruction`.
    member_images: np.ndarray
        Rendered members with shape
        `(number_of_members, image_height, image_width)`.
    coordinates: jax.Array
        Flattened image coordinates.
    render_at_coordinates: Callable[[Any, jax.Array], jax.Array]
        Callable receiving `(parameters, coordinates)` and returning scalar
        intensities.
    config: UncertaintyEvaluationConfig
        Method settings.

    Returns:
    --------
    UncertaintyMaps
        Truth-independent reconstruction and uncertainty products.
    """
    images = np.asarray(member_images)
    reconstruction_array = np.asarray(reconstruction)
    if images.ndim != 3:
        raise ValueError(
            "`member_images` must have shape "
            "(number_of_members, image_height, image_width)."
        )
    if images.shape[1:] != forward_model.image_shape:
        raise ValueError(
            "Ensemble image shape must match the forward model image shape."
        )
    if reconstruction_array.shape != forward_model.image_shape:
        raise ValueError(
            "`reconstruction` shape must match the forward model."
        )
    selected_methods = set(config.methods)

    deformation_fisher_information = None
    deformation_posterior_covariance = None
    deformation_node_variance = None
    deformation_uncertainty = None
    parameter_laplace_standard_deviation = None
    ensemble_mean_image = None
    ensemble_standard_deviation = None
    ensemble_fourier_variance = None
    fourier_information = None
    fourier_data_blind_variance = None
    fourier_blind_mask = None
    blind_content = None

    if "deformation" in selected_methods:
        deformation_result = deformation_laplace(
            forward_model,
            render_at_coordinates,
            reference_parameters,
            coordinates,
            config.deformation_grid_size,
            prior_precision=(
                config.deformation_prior_scale
                / config.deformation_grid_size**2
            ),
            diagonal=config.deformation_diagonal,
            measurement_chunk_size=(
                config.deformation_measurement_chunk_size
            ),
            map_chunk_size=config.deformation_map_chunk_size,
            show_progress=config.deformation_show_progress,
        )

        deformation_fisher_information = deformation_result.fisher_information
        deformation_posterior_covariance = deformation_result.posterior_covariance
        deformation_node_variance = deformation_result.node_variance
        deformation_uncertainty = deformation_result.uncertainty_map

    if "parameter_laplace" in selected_methods:
        laplace_result = linearized_parameter_laplace(
            forward_model,
            render_at_coordinates,
            reference_parameters,
            coordinates,
            prior_precision=config.laplace_prior_precision,
            curvature=config.laplace_curvature,
            rank=config.laplace_rank,
            random_seed=config.laplace_random_seed,
        )

        parameter_laplace_standard_deviation = laplace_result.pixel_standard_deviation

    if "ensemble" in selected_methods:
        ensemble_result = ensemble_uncertainty(images)
        ensemble_mean_image = ensemble_result.mean_image
        ensemble_standard_deviation = ensemble_result.pixel_standard_deviation
        ensemble_fourier_variance = ensemble_result.fourier_variance

    if "fourier_data_blind" in selected_methods:
        fourier_result = fourier_data_blind_map(
            forward_model,
            prior_precision_fraction=config.fourier_prior_precision_fraction,
            blind_information_fraction=config.blind_information_fraction,
        )
        blind_content = masked_fourier_content(reconstruction_array, fourier_result.blind_mask)

        fourier_information = fourier_result.information
        fourier_data_blind_variance = fourier_result.variance
        fourier_blind_mask = fourier_result.blind_mask

    return UncertaintyMaps(
        number_of_measurements=forward_model.number_of_measurements,
        reconstruction=reconstruction_array,
        member_images=images,
        deformation_fisher_information=deformation_fisher_information,
        deformation_posterior_covariance=deformation_posterior_covariance,
        deformation_node_variance=deformation_node_variance,
        deformation_uncertainty=deformation_uncertainty,
        parameter_laplace_standard_deviation=parameter_laplace_standard_deviation,
        ensemble_mean_image=ensemble_mean_image,
        ensemble_standard_deviation=ensemble_standard_deviation,
        ensemble_fourier_variance=ensemble_fourier_variance,
        fourier_information=fourier_information,
        fourier_data_blind_variance=fourier_data_blind_variance,
        fourier_blind_mask=fourier_blind_mask,
        blind_content=blind_content,
    )


def evaluate_uncertainty(
    maps: UncertaintyMaps,
    truth: np.ndarray,
    member_losses: tuple[float, ...],
    reference_member_index: int,
) -> UncertaintyEvaluationResult:
    """
    Evaluate precomputed uncertainty maps against known ground truth.

    Parameters:
    -----------
    maps: UncertaintyMaps
        Uncertainty maps for a given image.
    truth: np.ndarray
        Ground-truth image used for validation.
    member_losses: tuple[float, ...]
        Full-data loss for every rendered model member.
    reference_member_index: int
        Member selected as the reported reconstruction.

    Returns:
    --------
    UncertaintyEvaluationResult
        Truth-based errors, metrics, and storage-ready arrays.

    Raises:
    -------
    ValueError
        If truth, losses, or the selected member index are incompatible.
    """
    truth_array = np.asarray(truth)
    if truth_array.shape != maps.reconstruction.shape:
        raise ValueError("`truth` shape must match the reconstruction.")
    number_of_members = maps.member_images.shape[0]
    if len(member_losses) != number_of_members:
        raise ValueError(
            "`member_losses` length must match the ensemble size."
        )
    if not 0 <= reference_member_index < number_of_members:
        raise ValueError(
            "`reference_member_index` must select an ensemble member."
        )

    pixel_error = np.abs(maps.reconstruction - truth_array)
    absolute_fourier_error, relative_fourier_error = fourier_error_maps(
        maps.reconstruction,
        truth_array,
    )
    image_uncertainties = {
        name: uncertainty
        for name, uncertainty in {
            "deformation": maps.deformation_uncertainty,
            "parameter_laplace": maps.parameter_laplace_standard_deviation,
            "ensemble": maps.ensemble_standard_deviation,
        }.items()
        if uncertainty is not None
    }
    fourier_uncertainties = {
        name: uncertainty
        for name, uncertainty in {
            "ensemble": maps.ensemble_fourier_variance,
        }.items()
        if uncertainty is not None
    }

    metrics: dict[str, Any] = {
        "number_of_measurements": maps.number_of_measurements,
        "number_of_members": number_of_members,
        "member_losses": [float(loss) for loss in member_losses],
        "map_member_index": reference_member_index,
        "map_loss": float(member_losses[reference_member_index]),
        "map_truth_correlation": np.corrcoef(
            maps.reconstruction.ravel(),
            truth_array.ravel(),
        )[0, 1],
    }
    if maps.fourier_blind_mask is not None:
        metrics["data_blind_mode_fraction"] = float(maps.fourier_blind_mask.mean())
        metrics["reconstruction_power_on_blind_modes"] = (
            masked_fourier_power_fraction(
                maps.reconstruction,
                maps.fourier_blind_mask,
            )
        )
        if np.any(maps.fourier_blind_mask):
            metrics["mean_relative_error_on_data_blind_modes"] = float(
                relative_fourier_error[maps.fourier_blind_mask].mean()
            )
        sampled_mask = ~maps.fourier_blind_mask
        if np.any(sampled_mask):
            metrics["mean_relative_error_on_sampled_modes"] = float(
                relative_fourier_error[sampled_mask].mean()
            )
        metrics["blind_content_correlation"] = (
            masked_fourier_content_correlation(
                maps.reconstruction,
                truth_array,
                maps.fourier_blind_mask,
            )
        )
    for name, uncertainty in image_uncertainties.items():
        metrics[f"image/{name}/ause"] = area_under_sparsification_error(
            uncertainty,
            pixel_error,
        )
        metrics[f"image/{name}/spearman"] = spearman_rank_correlation(
            uncertainty,
            pixel_error,
        )
    for name, uncertainty in fourier_uncertainties.items():
        metrics[f"fourier_absolute/{name}/ause"] = (
            area_under_sparsification_error(uncertainty, absolute_fourier_error)
        )
        metrics[f"fourier_absolute/{name}/spearman"] = (
            spearman_rank_correlation(uncertainty, absolute_fourier_error)
        )
        metrics[f"fourier_relative/{name}/ause"] = (
            area_under_sparsification_error(uncertainty, relative_fourier_error)
        )
        metrics[f"fourier_relative/{name}/spearman"] = (
            spearman_rank_correlation(uncertainty, relative_fourier_error)
        )

    return UncertaintyEvaluationResult(
        maps=maps,
        truth=truth_array,
        pixel_error=pixel_error,
        absolute_fourier_error=absolute_fourier_error,
        relative_fourier_error=relative_fourier_error,
        metrics=metrics,
    )
