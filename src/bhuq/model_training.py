"""Train and load one consistent in-memory model representation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
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
from .models import CoordinateConvention
from .training import (
    ParameterTree,
    TrainingConfig,
    fit_ensemble,
)

LOGGER = logging.getLogger(__name__)


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
    model: nn.Module
        Flax module used to render every member.
    coordinates: jax.Array
        Default image-coordinate grid used during training.
    coordinate_convention: CoordinateConvention
        Ordering and endpoint convention represented by `coordinates`.
    members: tuple[TrainedModelMember, ...]
        Fitted members in seed order.
    image_shape: tuple[int, int]
        Image dimensions represented by every member.
    training_started_at: datetime
        UTC time at which optimization began.
    training_elapsed_seconds: float
        Wall-clock duration of the training operation.
    saved_model: SavedModel | None
        Persistent cache identity, or `None` when saving was disabled.
    """

    model: nn.Module
    coordinates: jax.Array
    coordinate_convention: CoordinateConvention
    members: tuple[TrainedModelMember, ...]
    image_shape: tuple[int, int]
    training_started_at: datetime
    training_elapsed_seconds: float
    saved_model: SavedModel | None

    @property
    def model_class(self) -> str:
        """Fully qualified class name of the runtime model."""
        return _qualified_model_class(self.model)

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
    coordinate_convention: CoordinateConvention
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
    coordinate_convention: CoordinateConvention,
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
    coordinate_convention: CoordinateConvention
        Ordering and endpoint convention represented by `coordinates`.
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
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("`seeds` must not contain duplicates.")
    if save and model_name is None:
        raise ValueError("`model_name` is required when `save=True`.")

    LOGGER.info(
        "JAX backend: %s; devices: %s",
        jax.default_backend(),
        jax.devices(),
    )
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

    trained_model = TrainedModel(
        model=model,
        coordinates=coordinates,
        coordinate_convention=coordinate_convention,
        members=members,
        image_shape=problem.image_shape,
        training_started_at=training_started_at,
        training_elapsed_seconds=elapsed_seconds,
        saved_model=None,
    )
    if not save:
        return trained_model
    return save_trained_model(
        trained_model,
        model_name=model_name,
        training_config=config,
        model_metadata=model_metadata,
        problem_metadata=problem_metadata,
        input_paths=input_paths,
        cache_root=cache_root,
    )


def save_trained_model(
    trained_model: TrainedModel,
    *,
    model_name: str,
    training_config: TrainingConfig,
    model_metadata: Any,
    problem_metadata: Any,
    input_paths: dict[str, str | Path] | None = None,
    cache_root: str | Path = Path("model_cache"),
) -> TrainedModel:
    """
    Save an already-trained ensemble in the standard model-cache format.

    Parameters:
    -----------
    trained_model: TrainedModel
        In-memory members and training diagnostics without a saved identity.
    model_name: str
        Semantic cache path.
    training_config: TrainingConfig
        Optimization settings shared by the members.
    model_metadata: Any
        Architecture configuration stored with the model.
    problem_metadata: Any
        Forward-problem configuration stored with the model.
    input_paths: dict[str, str | Path] | None
        Optional source-file override. By default, paths come from
        `problem_metadata.input_paths`.
    cache_root: str | Path
        Parent directory for saved model bundles.

    Returns:
    --------
    TrainedModel
        The same ensemble with its immutable saved-model identity attached.

    Raises:
    -------
    ValueError
        If the ensemble is empty, already saved, or contains duplicate seeds.
    """
    if trained_model.saved_model is not None:
        raise ValueError("`trained_model` is already saved.")
    if not trained_model.members:
        raise ValueError("`trained_model` must contain at least one member.")
    seeds = trained_model.seeds
    if len(set(seeds)) != len(seeds):
        raise ValueError("Trained member seeds must be unique.")

    metadata = _ModelTrainingMetadata(
        model_class=trained_model.model_class,
        image_shape=trained_model.image_shape,
        coordinate_convention=trained_model.coordinate_convention,
        model=model_metadata,
        problem=problem_metadata,
        optimization=training_config,
    )
    saved_model = _save_model(
        model_name=model_name,
        parameters=trained_model.parameters,
        seeds=seeds,
        final_losses=tuple(
            member.final_loss for member in trained_model.members
        ),
        recorded_steps=tuple(
            member.recorded_steps for member in trained_model.members
        ),
        loss_histories=tuple(
            member.loss_history for member in trained_model.members
        ),
        training_metadata=metadata,
        input_paths=(
            _metadata_input_paths(problem_metadata)
            if input_paths is None
            else input_paths
        ),
        training_started_at=trained_model.training_started_at,
        training_elapsed_seconds=trained_model.training_elapsed_seconds,
        cache_root=cache_root,
    )
    return replace(trained_model, saved_model=saved_model)


def load_model(
    saved_model: SavedModel,
    model: nn.Module,
    *,
    input_paths: dict[str, str | Path] | None = None,
) -> TrainedModel:
    """
    Load a saved model into the same representation returned by training.

    Parameters:
    -----------
    saved_model: SavedModel
        A model-cache record.
    model: nn.Module
        Constructed model matching the saved model class.
    input_paths: dict[str, str | Path] | None
        Optional relocated inputs. Recorded paths are used by default.

    Returns:
    --------
    TrainedModel
        Loaded parameter trees, diagnostics, metadata, and saved identity.

    Raises:
    -------
    ValueError
        If the model class, input files, or histories differ
        from the saved record.
    """
    if _qualified_model_class(model) != saved_model.model_class:
        raise ValueError(
            "The runtime model class differs from the saved model."
        )
    validate_model_inputs(
        saved_model,
        saved_model.input_paths if input_paths is None else input_paths,
    )
    coordinates = saved_model.build_coordinates()

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
        model=model,
        coordinates=coordinates,
        coordinate_convention=saved_model.coordinate_convention,
        members=tuple(members),
        image_shape=saved_model.image_shape,
        training_started_at=datetime.fromisoformat(
            saved_model.provenance["training_started_at"]
        ),
        training_elapsed_seconds=float(
            saved_model.provenance["training_elapsed_seconds"]
        ),
        saved_model=saved_model,
    )


def _metadata_input_paths(metadata: Any) -> dict[str, Path]:
    """Return source paths declared by problem metadata."""
    input_paths = getattr(metadata, "input_paths", None)
    if input_paths is None:
        return {}
    if not isinstance(input_paths, dict):
        raise TypeError("`problem_metadata.input_paths` must be a dictionary.")
    return {
        str(name): Path(path)
        for name, path in input_paths.items()
    }


def _qualified_model_class(model: nn.Module) -> str:
    """Return the fully qualified class name used for compatibility checks."""
    model_class = type(model)
    return f"{model_class.__module__}.{model_class.__qualname__}"
