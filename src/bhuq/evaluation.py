"""uncertainty evaluation for fitted two-dimensional image models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import numpy as np

from .forward import LinearInverseProblem
from .metrics import (
    SparsificationResult,
    fourier_error_maps,
    masked_fourier_content,
    masked_fourier_content_correlation,
    masked_fourier_power_fraction,
    sparsification_curve,
    spearman_rank_correlation,
)
from .uq.deformation import DeformationLaplaceResult, deformation_laplace
from .uq.ensemble import EnsembleUncertainty, ensemble_uncertainty
from .uq.fourier import FourierDataBlindResult, fourier_data_blind_map
from .uq.laplace import (
    LaplaceCurvature,
    LinearizedLaplaceResult,
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
class UncertaintyResults:
    """
    Store truth-independent outputs from selected uncertainty methods.

    Parameters:
    -----------
    number_of_measurements: int
        Number of measurements used to calculate uncertainty.
    reconstruction: np.ndarray
        Selected reconstruction.
    deformation: DeformationLaplaceResult | None
        Coordinate-deformation Laplace result, or `None` when not requested.
    parameter_laplace: LinearizedLaplaceResult | None
        Parameter-space Laplace result, or `None` when not requested.
    ensemble: EnsembleUncertainty | None
        Ensemble uncertainty result, or `None` when not requested.
    fourier_data_blind: FourierDataBlindResult | None
        Fourier identifiability result, or `None` when not requested.
    blind_content: np.ndarray | None
        Reconstruction content occupying data-blind Fourier modes, or `None`
        when Fourier identifiability was not requested.
    """

    number_of_measurements: int
    reconstruction: np.ndarray
    deformation: DeformationLaplaceResult | None
    parameter_laplace: LinearizedLaplaceResult | None
    ensemble: EnsembleUncertainty | None
    fourier_data_blind: FourierDataBlindResult | None
    blind_content: np.ndarray | None

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Flatten numerical method products for artifact serialization."""
        arrays = {
            "number_of_measurements": np.asarray(
                self.number_of_measurements
            ),
            "reconstruction": self.reconstruction,
        }
        if self.deformation is not None:
            arrays.update(
                {
                    "deformation_fisher_information": (
                        self.deformation.fisher_information
                    ),
                    "deformation_posterior_covariance": (
                        self.deformation.posterior_covariance
                    ),
                    "deformation_node_variance": (
                        self.deformation.node_variance
                    ),
                    "deformation_uncertainty": (
                        self.deformation.uncertainty_map
                    ),
                    "deformation_diagonal": np.asarray(
                        self.deformation.diagonal
                    ),
                }
            )
        if self.parameter_laplace is not None:
            arrays.update(
                {
                    "parameter_laplace_pixel_standard_deviation": (
                        self.parameter_laplace.pixel_standard_deviation
                    ),
                    "parameter_laplace_prior_precision": np.asarray(
                        self.parameter_laplace.prior_precision
                    ),
                    "parameter_laplace_curvature": np.asarray(
                        self.parameter_laplace.curvature
                    ),
                    "parameter_laplace_parameter_count": np.asarray(
                        self.parameter_laplace.parameter_count
                    ),
                }
            )
            if self.parameter_laplace.rank is not None:
                arrays["parameter_laplace_rank"] = np.asarray(
                    self.parameter_laplace.rank
                )
        if self.ensemble is not None:
            arrays.update(
                {
                    "ensemble_mean_image": self.ensemble.mean_image,
                    "ensemble_pixel_standard_deviation": (
                        self.ensemble.pixel_standard_deviation
                    ),
                    "ensemble_fourier_variance": (
                        self.ensemble.fourier_variance
                    ),
                }
            )
        if self.fourier_data_blind is not None:
            arrays.update(
                {
                    "fourier_information": (
                        self.fourier_data_blind.information
                    ),
                    "fourier_data_blind_variance": (
                        self.fourier_data_blind.variance
                    ),
                    "fourier_blind_mask": (
                        self.fourier_data_blind.blind_mask
                    ),
                    "fourier_blind_fraction": np.asarray(
                        self.fourier_data_blind.blind_fraction
                    ),
                    "fourier_prior_precision": np.asarray(
                        self.fourier_data_blind.prior_precision
                    ),
                }
            )
        if self.blind_content is not None:
            arrays["blind_content"] = self.blind_content
        return arrays

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, np.ndarray],
    ) -> UncertaintyResults:
        """
        Reconstruct nested uncertainty results from serialized arrays.

        Parameters:
        -----------
        arrays: dict[str, np.ndarray]
            Arrays produced by the `arrays` property.

        Returns:
        --------
        UncertaintyResults
            Reconstructed truth-independent method results.
        """
        deformation = None
        if "deformation_uncertainty" in arrays:
            deformation = DeformationLaplaceResult(
                fisher_information=np.asarray(
                    arrays["deformation_fisher_information"]
                ),
                posterior_covariance=np.asarray(
                    arrays["deformation_posterior_covariance"]
                ),
                node_variance=np.asarray(
                    arrays["deformation_node_variance"]
                ),
                uncertainty_map=np.asarray(
                    arrays["deformation_uncertainty"]
                ),
                diagonal=bool(
                    np.asarray(arrays["deformation_diagonal"]).item()
                ),
            )

        parameter_laplace = None
        if "parameter_laplace_pixel_standard_deviation" in arrays:
            rank = None
            if "parameter_laplace_rank" in arrays:
                rank = int(
                    np.asarray(arrays["parameter_laplace_rank"]).item()
                )
            parameter_laplace = LinearizedLaplaceResult(
                pixel_standard_deviation=np.asarray(
                    arrays[
                        "parameter_laplace_pixel_standard_deviation"
                    ]
                ),
                prior_precision=float(
                    np.asarray(
                        arrays["parameter_laplace_prior_precision"]
                    ).item()
                ),
                curvature=str(
                    np.asarray(
                        arrays["parameter_laplace_curvature"]
                    ).item()
                ),
                parameter_count=int(
                    np.asarray(
                        arrays["parameter_laplace_parameter_count"]
                    ).item()
                ),
                rank=rank,
            )

        ensemble = None
        if "ensemble_mean_image" in arrays:
            ensemble = EnsembleUncertainty(
                mean_image=np.asarray(arrays["ensemble_mean_image"]),
                pixel_standard_deviation=np.asarray(
                    arrays["ensemble_pixel_standard_deviation"]
                ),
                fourier_variance=np.asarray(
                    arrays["ensemble_fourier_variance"]
                ),
            )

        fourier_data_blind = None
        if "fourier_information" in arrays:
            fourier_data_blind = FourierDataBlindResult(
                information=np.asarray(arrays["fourier_information"]),
                variance=np.asarray(
                    arrays["fourier_data_blind_variance"]
                ),
                blind_mask=np.asarray(
                    arrays["fourier_blind_mask"],
                    dtype=bool,
                ),
                blind_fraction=float(
                    np.asarray(arrays["fourier_blind_fraction"]).item()
                ),
                prior_precision=float(
                    np.asarray(arrays["fourier_prior_precision"]).item()
                ),
            )

        blind_content = None
        if "blind_content" in arrays:
            blind_content = np.asarray(arrays["blind_content"])

        return cls(
            number_of_measurements=int(
                np.asarray(arrays["number_of_measurements"]).item()
            ),
            reconstruction=np.asarray(arrays["reconstruction"]),
            deformation=deformation,
            parameter_laplace=parameter_laplace,
            ensemble=ensemble,
            fourier_data_blind=fourier_data_blind,
            blind_content=blind_content,
        )


@dataclass(frozen=True)
class UncertaintyEvaluationResult:
    """
    Parameters:
    -----------
    results: UncertaintyResults
        Truth-independent uncertainty method results.
    truth: np.ndarray
        Ground-truth image.
    pixel_error: np.ndarray
        Absolute image-space reconstruction error.
    absolute_fourier_error: np.ndarray
        Absolute Fourier-domain reconstruction error.
    relative_fourier_error: np.ndarray
        Relative Fourier-domain reconstruction error.
    sparsification_results: dict[str, SparsificationResult]
        Full sparsification curves keyed by evaluation domain and method.
    metrics: dict[str, Any]
        Scalar and structured validation metrics.
    """

    results: UncertaintyResults
    truth: np.ndarray
    pixel_error: np.ndarray
    absolute_fourier_error: np.ndarray
    relative_fourier_error: np.ndarray
    sparsification_results: dict[str, SparsificationResult]
    metrics: dict[str, Any]

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Flatten uncertainty and truth-based products for serialization."""
        arrays = {
            **self.results.arrays,
            "truth": self.truth,
            "pixel_error": self.pixel_error,
            "absolute_fourier_error": self.absolute_fourier_error,
            "relative_fourier_error": self.relative_fourier_error,
        }
        for name, result in self.sparsification_results.items():
            prefix = f"sparsification_{name.replace('/', '_')}"
            arrays[f"{prefix}_removed_fractions"] = result.removed_fractions
            arrays[f"{prefix}_uncertainty_ranked_error"] = (
                result.uncertainty_ranked_error
            )
            arrays[f"{prefix}_oracle_ranked_error"] = (
                result.oracle_ranked_error
            )
            arrays[f"{prefix}_sparsification_error"] = (
                result.sparsification_error
            )
            arrays[f"{prefix}_area_under_sparsification_error"] = np.asarray(
                result.area_under_sparsification_error
            )
        arrays["sparsification_result_names"] = np.asarray(
            tuple(self.sparsification_results)
        )
        return arrays

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, np.ndarray],
        *,
        metrics: dict[str, Any],
    ) -> UncertaintyEvaluationResult:
        """
        Reconstruct an evaluated uncertainty result from a run artifact.

        Parameters:
        -----------
        arrays: dict[str, np.ndarray]
            Arrays produced by the `arrays` property.
        metrics: dict[str, Any]
            Evaluation metrics loaded from the run's `metrics.json`.

        Returns:
        --------
        UncertaintyEvaluationResult
            Reconstructed nested method results, errors, curves, and metrics.
        """
        sparsification_results = {}
        result_names = np.asarray(
            arrays["sparsification_result_names"]
        ).reshape(-1)
        for stored_name in result_names:
            name = str(stored_name)
            prefix = f"sparsification_{name.replace('/', '_')}"
            sparsification_results[name] = SparsificationResult(
                removed_fractions=np.asarray(
                    arrays[f"{prefix}_removed_fractions"]
                ),
                uncertainty_ranked_error=np.asarray(
                    arrays[f"{prefix}_uncertainty_ranked_error"]
                ),
                oracle_ranked_error=np.asarray(
                    arrays[f"{prefix}_oracle_ranked_error"]
                ),
                area_under_sparsification_error=float(
                    np.asarray(
                        arrays[
                            f"{prefix}_area_under_sparsification_error"
                        ]
                    ).item()
                ),
            )

        return cls(
            results=UncertaintyResults.from_arrays(arrays),
            truth=np.asarray(arrays["truth"]),
            pixel_error=np.asarray(arrays["pixel_error"]),
            absolute_fourier_error=np.asarray(
                arrays["absolute_fourier_error"]
            ),
            relative_fourier_error=np.asarray(
                arrays["relative_fourier_error"]
            ),
            sparsification_results=sparsification_results,
            metrics=dict(metrics),
        )


def compute_uncertainty_results(
    *,
    forward_model: LinearInverseProblem,
    reconstruction: np.ndarray,
    reference_parameters: Any,
    member_images: np.ndarray,
    coordinates: jax.Array,
    render_at_coordinates: Callable[[Any, jax.Array], jax.Array],
    config: UncertaintyEvaluationConfig,
) -> UncertaintyResults:
    """
    Compute selected uncertainty method results without requiring ground truth.

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
    UncertaintyResults
        Truth-independent reconstruction and method results.
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

    deformation_result = None
    parameter_laplace_result = None
    fourier_result = None
    ensemble_result = None
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
            measurement_chunk_size=config.deformation_measurement_chunk_size,
            map_chunk_size=config.deformation_map_chunk_size,
            show_progress=config.deformation_show_progress,
        )

    if "parameter_laplace" in selected_methods:
        parameter_laplace_result = linearized_parameter_laplace(
            forward_model,
            render_at_coordinates,
            reference_parameters,
            coordinates,
            prior_precision=config.laplace_prior_precision,
            curvature=config.laplace_curvature,
            rank=config.laplace_rank,
            random_seed=config.laplace_random_seed,
        )

    if "ensemble" in selected_methods:
        ensemble_result = ensemble_uncertainty(images)

    if "fourier_data_blind" in selected_methods:
        fourier_result = fourier_data_blind_map(
            forward_model,
            prior_precision_fraction=config.fourier_prior_precision_fraction,
            blind_information_fraction=config.blind_information_fraction,
        )
        blind_content = masked_fourier_content(
            reconstruction_array,
            fourier_result.blind_mask,
        )

    return UncertaintyResults(
        number_of_measurements=forward_model.number_of_measurements,
        reconstruction=reconstruction_array,
        deformation=deformation_result,
        parameter_laplace=parameter_laplace_result,
        ensemble=ensemble_result,
        fourier_data_blind=fourier_result,
        blind_content=blind_content,
    )


def evaluate_uncertainty(
    results: UncertaintyResults,
    truth: np.ndarray,
    member_losses: tuple[float, ...],
    reference_member_index: int,
) -> UncertaintyEvaluationResult:
    """
    Evaluate precomputed uncertainty results against known ground truth.

    Parameters:
    -----------
    results: UncertaintyResults
        Truth-independent uncertainty method results.
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
    if truth_array.shape != results.reconstruction.shape:
        raise ValueError("`truth` shape must match the reconstruction.")
    number_of_members = len(member_losses)
    if not 0 <= reference_member_index < number_of_members:
        raise ValueError(
            "`reference_member_index` must select an ensemble member."
        )

    pixel_error = np.abs(results.reconstruction - truth_array)
    absolute_fourier_error, relative_fourier_error = fourier_error_maps(
        results.reconstruction,
        truth_array,
    )
    image_uncertainties: dict[str, np.ndarray] = {}
    if results.deformation is not None:
        image_uncertainties["deformation"] = (
            results.deformation.uncertainty_map
        )
    if results.parameter_laplace is not None:
        image_uncertainties["parameter_laplace"] = (
            results.parameter_laplace.pixel_standard_deviation
        )
    if results.ensemble is not None:
        image_uncertainties["ensemble"] = (
            results.ensemble.pixel_standard_deviation
        )

    fourier_uncertainties: dict[str, np.ndarray] = {}
    if results.ensemble is not None:
        fourier_uncertainties["ensemble"] = results.ensemble.fourier_variance

    metrics: dict[str, Any] = {
        "number_of_measurements": results.number_of_measurements,
        "number_of_members": number_of_members,
        "member_losses": [float(loss) for loss in member_losses],
        "map_member_index": reference_member_index,
        "map_loss": float(member_losses[reference_member_index]),
        "map_truth_correlation": np.corrcoef(
            results.reconstruction.ravel(),
            truth_array.ravel(),
        )[0, 1],
    }
    if results.fourier_data_blind is not None:
        blind_mask = results.fourier_data_blind.blind_mask
        metrics["data_blind_mode_fraction"] = (
            results.fourier_data_blind.blind_fraction
        )
        metrics["reconstruction_power_on_blind_modes"] = (
            masked_fourier_power_fraction(
                results.reconstruction,
                blind_mask,
            )
        )
        if np.any(blind_mask):
            metrics["mean_relative_error_on_data_blind_modes"] = float(
                relative_fourier_error[blind_mask].mean()
            )
        sampled_mask = ~blind_mask
        if np.any(sampled_mask):
            metrics["mean_relative_error_on_sampled_modes"] = float(
                relative_fourier_error[sampled_mask].mean()
            )
        metrics["blind_content_correlation"] = (
            masked_fourier_content_correlation(
                results.reconstruction,
                truth_array,
                blind_mask,
            )
        )
    sparsification_results: dict[str, SparsificationResult] = {}
    for name, uncertainty in image_uncertainties.items():
        result_name = f"image/{name}"
        sparsification_result = sparsification_curve(
            uncertainty,
            pixel_error,
        )
        sparsification_results[result_name] = sparsification_result
        metrics[f"{result_name}/ause"] = (
            sparsification_result.area_under_sparsification_error
        )
        metrics[f"image/{name}/spearman"] = spearman_rank_correlation(
            uncertainty,
            pixel_error,
        )
    for name, uncertainty in fourier_uncertainties.items():
        absolute_result_name = f"fourier_absolute/{name}"
        absolute_sparsification = sparsification_curve(
            uncertainty,
            absolute_fourier_error,
        )
        sparsification_results[absolute_result_name] = absolute_sparsification
        metrics[f"{absolute_result_name}/ause"] = (
            absolute_sparsification.area_under_sparsification_error
        )
        metrics[f"fourier_absolute/{name}/spearman"] = (
            spearman_rank_correlation(uncertainty, absolute_fourier_error)
        )
        relative_result_name = f"fourier_relative/{name}"
        relative_sparsification = sparsification_curve(
            uncertainty,
            relative_fourier_error,
        )
        sparsification_results[relative_result_name] = relative_sparsification
        metrics[f"{relative_result_name}/ause"] = (
            relative_sparsification.area_under_sparsification_error
        )
        metrics[f"fourier_relative/{name}/spearman"] = (
            spearman_rank_correlation(uncertainty, relative_fourier_error)
        )

    return UncertaintyEvaluationResult(
        results=results,
        truth=truth_array,
        pixel_error=pixel_error,
        absolute_fourier_error=absolute_fourier_error,
        relative_fourier_error=relative_fourier_error,
        sparsification_results=sparsification_results,
        metrics=metrics,
    )
