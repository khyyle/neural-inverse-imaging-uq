"""Apply uncertainty evaluation to a trained model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from flax import linen as nn

from .evaluation import (
    UncertaintyEvaluationConfig,
    UncertaintyEvaluationResult,
    compute_uncertainty_maps,
    evaluate_uncertainty,
)
from .forward import LinearInverseProblem
from .model_cache import SavedModel
from .model_training import TrainedModel


@dataclass(frozen=True)
class ModelEvaluationResult:
    """
    Parameters:
    -----------
    uncertainty: UncertaintyEvaluationResult
        Numerical arrays and metrics.
    config: UncertaintyEvaluationConfig
        Methods and hyperparameters used to produce the result.
    saved_model: SavedModel | None
        Persistent model identity, or `None` for unsaved training.
    """
    uncertainty: UncertaintyEvaluationResult
    config: UncertaintyEvaluationConfig
    saved_model: SavedModel | None

    @property
    def metrics(self) -> dict[str, Any]:
        """Return named scalar and structured evaluation metrics."""
        return self.uncertainty.metrics

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Return named numerical products for saving or plotting."""
        return self.uncertainty.arrays


def evaluate_model(
    trained_model: TrainedModel,
    model: nn.Module,
    coordinates: jax.Array,
    problem: LinearInverseProblem,
    truth: np.ndarray,
    config: UncertaintyEvaluationConfig,
) -> ModelEvaluationResult:
    """
    Apply selected UQ methods to one trained model representation.

    This function accepts the same `TrainedModel` type from `train_model()` and
    `load_model()`. It performs no filesystem writes.

    Parameters:
    -----------
    trained_model: TrainedModel
        Fitted parameter trees and optional saved identity.
    model: nn.Module
        Caller-constructed Flax model matching `trained_model`.
    coordinates: jax.Array
        Coordinates used to render the model.
    problem: LinearInverseProblem
        Measurement operator, observations, noise, and image shape.
    truth: np.ndarray
        Ground-truth image used to compute validation errors.
    config: UncertaintyEvaluationConfig
        Selected UQ methods and numerical settings.

    Returns:
    --------
    ModelEvaluationResult
        Numerical UQ products coupled to model and method provenance.

    Raises:
    -------
    ValueError
        If the runtime model or image shape is incompatible or no members
        exist.
    """
    model_class = type(model)
    qualified_model_class = (
        f"{model_class.__module__}.{model_class.__qualname__}"
    )
    if trained_model.model_class != qualified_model_class:
        raise ValueError(
            "The runtime model class differs from the trained model."
        )
    if trained_model.image_shape != problem.image_shape:
        raise ValueError(
            "The inverse-problem image shape differs from the trained model."
        )
    if not trained_model.members:
        raise ValueError("`trained_model` must contain at least one member.")

    parameters = trained_model.parameters
    member_images = np.stack(
        [
            np.asarray(
                model.apply(
                    {"params": member_parameters},
                    coordinates,
                )
            ).reshape(problem.image_shape)
            for member_parameters in parameters
        ]
    )
    member_losses = tuple(
        float(problem.gaussian_negative_log_likelihood(image))
        for image in member_images
    )
    best_member_index = int(np.argmin(member_losses))

    def render_at_coordinates(
        member_parameters: Any,
        sample_coordinates: jax.Array,
    ) -> jax.Array:
        return model.apply(
            {"params": member_parameters},
            sample_coordinates,
        )

    maps = compute_uncertainty_maps(
        forward_model=problem,
        reconstruction=member_images[best_member_index],
        reference_parameters=parameters[best_member_index],
        member_images=member_images,
        coordinates=coordinates,
        render_at_coordinates=render_at_coordinates,
        config=config,
    )
    uncertainty = evaluate_uncertainty(
        maps,
        truth,
        member_losses,
        best_member_index,
    )
    return ModelEvaluationResult(
        uncertainty=uncertainty,
        config=config,
        saved_model=trained_model.saved_model,
    )
