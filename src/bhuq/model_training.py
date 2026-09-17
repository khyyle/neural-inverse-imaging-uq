"""Train and load one consistent in-memory model representation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import linen as nn

from .forward import LinearInverseProblem
from .model_cache import (
    SavedModel,
    _save_model,
    validate_model_inputs,
)
from .training import (
    ParameterTree,
    TrainingConfig,
    fit_ensemble,
)


@dataclass(frozen=True)
class TrainedModelMember:
    """
    Store one fitted parameter tree and its optimization history.

    Parameters:
    -----------
    seed: int
        Initialization and batching seed.
    parameters: ParameterTree
        Final fitted Flax parameters.
    final_loss: float
        Full-data loss after training.
    recorded_steps: np.ndarray
        Optimizer steps represented in `loss_history`.
    loss_history: np.ndarray
        Recorded batch losses.
    """
    seed: int
    parameters: ParameterTree
    final_loss: float
    recorded_steps: np.ndarray
    loss_history: np.ndarray


@dataclass(frozen=True)
class TrainedModel:
    """
    Represent the same usable model whether freshly trained or loaded.

    Parameters:
    -----------
    members: tuple[TrainedModelMember, ...]
        Fitted members in seed order.
    model_class: str
        Fully qualified Flax model class.
    image_shape: tuple[int, int]
        Image dimensions represented by every member.
    training_started_at: datetime
        UTC time at which optimization began.
    training_elapsed_seconds: float
        Wall-clock duration of the training operation.
    saved_model: SavedModel | None
        Persistent cache identity, or `None` when saving was disabled.
    """

    members: tuple[TrainedModelMember, ...]
    model_class: str
    image_shape: tuple[int, int]
    training_started_at: datetime
    training_elapsed_seconds: float
    saved_model: SavedModel | None

    @property
    def parameters(self) -> tuple[ParameterTree, ...]:
        """Return fitted parameter trees in member order."""
        return tuple(member.parameters for member in self.members)

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return training seeds in member order."""
        return tuple(member.seed for member in self.members)


@dataclass(frozen=True)
class _ModelTrainingMetadata:
    model_class: str
    image_shape: tuple[int, int]
    coordinate_shape: tuple[int, ...]
    model: Any
    problem: Any
    optimization: TrainingConfig


def train_model(
    model: nn.Module,
    coordinates: jax.Array,
    problem: LinearInverseProblem,
    config: TrainingConfig,
    seeds: tuple[int, ...],
    *,
    save: bool = True,
    model_name: str | None = None,
    model_metadata: Any = None,
    problem_metadata: Any = None,
    input_paths: dict[str, str | Path] | None = None,
    cache_root: str | Path = Path("model_cache"),
) -> TrainedModel:
    """
    Train an ensemble and save its complete model record.

    Pass `save=False` to avoid writing to the `model_cache`. When saving is enabled,
    `model_name` provides a semantic cache path.

    Parameters:
    -----------
    model: nn.Module
        Caller-constructed Flax image model.
    coordinates: jax.Array
        Coordinates used to initialize and render the model.
    problem: LinearInverseProblem
        Caller-constructed measurement problem.
    config: TrainingConfig
        Optimizer and training-loop settings.
    seeds: tuple[int, ...]
        Independent ensemble seeds.
    save: bool
        Whether to save parameters, histories, timing, and provenance.
    model_name: str | None
        Semantic cache path. Required when `save=True`.
    model_metadata: Any
        Architecture configuration stored with a saved model.
    problem_metadata: Any
        Forward-problem configuration stored with a saved model.
    input_paths: dict[str, str | Path] | None
        Source files hashed into saved-model provenance.
    cache_root: str | Path
        Parent directory for saved models.

    Returns:
    --------
    TrainedModel
        Usable in-memory model with optional saved identity.

    Raises:
    -------
    ValueError
        If seeds are empty or saving lacks a model name.
    """
    resolved_seeds = tuple(seeds)
    if not resolved_seeds:
        raise ValueError("`seeds` must contain at least one value.")
    if save and model_name is None:
        raise ValueError("`model_name` is required when `save=True`.")

    model_class = _qualified_model_class(model)
    training_started_at = datetime.now(UTC)
    start_time = time.monotonic()
    training_results = fit_ensemble(
        model,
        coordinates,
        problem,
        config,
        resolved_seeds,
    )
    elapsed_seconds = time.monotonic() - start_time
    members = tuple(
        TrainedModelMember(
            seed=seed,
            parameters=result.parameters,
            final_loss=result.final_loss,
            recorded_steps=result.recorded_steps,
            loss_history=result.loss_history,
        )
        for seed, result in zip(
            resolved_seeds,
            training_results,
            strict=True,
        )
    )

    saved_model = None
    if save:
        metadata = _ModelTrainingMetadata(
            model_class=model_class,
            image_shape=problem.image_shape,
            coordinate_shape=tuple(coordinates.shape),
            model=model_metadata,
            problem=problem_metadata,
            optimization=config,
        )
        saved_model = _save_model(
            model_name=model_name,
            parameters=tuple(member.parameters for member in members),
            seeds=resolved_seeds,
            final_losses=tuple(member.final_loss for member in members),
            recorded_steps=tuple(
                member.recorded_steps for member in members
            ),
            loss_histories=tuple(
                member.loss_history for member in members
            ),
            training_metadata=metadata,
            input_paths=input_paths or {},
            training_started_at=training_started_at,
            training_elapsed_seconds=elapsed_seconds,
            cache_root=cache_root,
        )

    return TrainedModel(
        members=members,
        model_class=model_class,
        image_shape=problem.image_shape,
        training_started_at=training_started_at,
        training_elapsed_seconds=elapsed_seconds,
        saved_model=saved_model,
    )


def load_model(
    saved_model: SavedModel,
    model: nn.Module,
    coordinates: jax.Array,
    *,
    input_paths: dict[str, str | Path],
) -> TrainedModel:
    """
    Load a saved model into the same representation returned by training.

    Parameters:
    -----------
    saved_model: SavedModel
        A model-cache record.
    model: nn.Module
        Constructed model matching the saved model class.
    coordinates: jax.Array
        Coordinates matching the saved training coordinate shape.
    input_paths: dict[str, str | Path]
        Current input files verified against training provenance.

    Returns:
    --------
    TrainedModel
        Loaded parameter trees, diagnostics, metadata, and saved identity.

    Raises:
    -------
    ValueError
        If the model class, coordinate shape, input files, or histories differ
        from the saved record.
    """
    if _qualified_model_class(model) != saved_model.model_class:
        raise ValueError(
            "The runtime model class differs from the saved model."
        )
    coordinate_shape = tuple(
        int(size)
        for size in saved_model.training_metadata["coordinate_shape"]
    )
    if tuple(coordinates.shape) != coordinate_shape:
        raise ValueError(
            "Runtime coordinates differ from the saved training shape."
        )
    validate_model_inputs(saved_model, input_paths)

    template_parameters = model.init(
        jax.random.PRNGKey(0),
        coordinates,
    )["params"]
    parameters = saved_model.load_parameters(template_parameters)
    histories = saved_model.training_history
    if len(parameters) != len(histories):
        raise ValueError(
            "Saved parameters and training histories have different lengths."
        )
    members = []
    for parameter_tree, manifest_seed, history in zip(
        parameters,
        saved_model.member_seeds,
        histories,
        strict=True,
    ):
        points = history.get("history")
        if not isinstance(points, list):
            raise ValueError(
                "Each saved member must contain a training history."
            )
        history_seed = int(history["seed"])
        if history_seed != manifest_seed:
            raise ValueError(
                "Saved checkpoint and training-history seeds differ."
            )
        members.append(
            TrainedModelMember(
                seed=manifest_seed,
                parameters=parameter_tree,
                final_loss=float(history["final_loss"]),
                recorded_steps=np.asarray(
                    [point["step"] for point in points],
                ),
                loss_history=np.asarray(
                    [point["loss"] for point in points],
                ),
            )
        )

    return TrainedModel(
        members=tuple(members),
        model_class=saved_model.model_class,
        image_shape=saved_model.image_shape,
        training_started_at=datetime.fromisoformat(
            saved_model.provenance["training_started_at"]
        ),
        training_elapsed_seconds=float(
            saved_model.provenance["training_elapsed_seconds"]
        ),
        saved_model=saved_model,
    )


def _qualified_model_class(model: nn.Module) -> str:
    """Return the fully qualified class name used for compatibility checks."""
    model_class = type(model)
    return f"{model_class.__module__}.{model_class.__qualname__}"
