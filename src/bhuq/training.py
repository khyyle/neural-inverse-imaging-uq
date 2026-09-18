"""Train Flax image models against linear inverse problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm.auto import tqdm

from .forward import LinearInverseProblem

type ParameterTree = Any
type LearningRateSchedule = Literal["constant", "linear", "cosine"]

SUPPORTED_LEARNING_RATE_SCHEDULES: tuple[LearningRateSchedule, ...] = (
    "constant",
    "linear",
    "cosine",
)


@dataclass(frozen=True)
class TrainingConfig:
    """
    Parameters:
    -----------
    number_of_steps: int
        Number of optimizer updates.
    initial_learning_rate: float
        Learning rate at the beginning of training.
    final_learning_rate: float
        Learning rate at the end of a linear or cosine schedule.
    learning_rate_schedule: LearningRateSchedule
        One of `constant`, `linear`, or `cosine`.
    batch_size: int | None
        Measurements sampled without replacement per update. `None` uses all
        measurements.
    log_interval: int
        Number of steps between recorded training losses.
    show_progress: bool
        Whether to display a tqdm progress bar.
    """

    number_of_steps: int = 10_000
    initial_learning_rate: float = 1e-3
    final_learning_rate: float = 1e-4
    learning_rate_schedule: LearningRateSchedule = "linear"
    batch_size: int | None = 501
    log_interval: int = 100
    show_progress: bool = True

    def __post_init__(self) -> None:
        """Reject invalid optimization settings before compilation."""
        if self.number_of_steps <= 0:
            raise ValueError("`number_of_steps` must be positive.")
        if self.initial_learning_rate <= 0.0:
            raise ValueError("`initial_learning_rate` must be positive.")
        if self.final_learning_rate <= 0.0:
            raise ValueError("`final_learning_rate` must be positive.")
        if self.learning_rate_schedule not in SUPPORTED_LEARNING_RATE_SCHEDULES:
            raise ValueError(
                f"`learning_rate_schedule` must be one of "
                f"{SUPPORTED_LEARNING_RATE_SCHEDULES}."
            )
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive or `None`.")
        if self.log_interval <= 0:
            raise ValueError("`log_interval` must be positive.")


@dataclass(frozen=True)
class TrainingResult:
    """
    Store fitted parameters and optimization diagnostics.

    Parameters:
    -----------
    state: train_state.TrainState
        Final model parameters, optimizer state, apply function, and step.
    final_loss: float
        Full-data Gaussian negative log-likelihood after training.
    recorded_steps: np.ndarray
        Optimizer steps represented in `loss_history`.
    loss_history: np.ndarray
        Batch losses recorded during optimization.
    """

    state: train_state.TrainState
    final_loss: float
    recorded_steps: np.ndarray
    loss_history: np.ndarray

    @property
    def parameters(self) -> ParameterTree:
        return self.state.params


def _build_learning_rate_schedule(config: TrainingConfig) -> optax.Schedule:
    """Construct the configured Optax learning-rate schedule."""
    if config.learning_rate_schedule == "constant":
        return optax.constant_schedule(config.initial_learning_rate)
    if config.learning_rate_schedule == "linear":
        return optax.linear_schedule(
            init_value=config.initial_learning_rate,
            end_value=config.final_learning_rate,
            transition_steps=config.number_of_steps,
        )
    final_fraction = (
        config.final_learning_rate / config.initial_learning_rate
    )
    return optax.cosine_decay_schedule(
        init_value=config.initial_learning_rate,
        decay_steps=config.number_of_steps,
        alpha=final_fraction,
    )


def fit_linear_inverse_problem(
    model: nn.Module,
    coordinates: jax.Array,
    problem: LinearInverseProblem,
    config: TrainingConfig,
    *,
    seed: int,
) -> TrainingResult:
    """
    Fit a Flax image model to linear measurements.

    Training renders an image from the current parameters and delegates the
    measurement likelihood to `problem`.

    Parameters:
    -----------
    model: nn.Module
        Flax module mapping coordinates to image values.
    coordinates: jax.Array
        Coordinates used to initialize and render the model.
    problem: LinearInverseProblem
        Fixed measurement operator, observations, and noise.
    config: TrainingConfig
        Optimization settings.
    seed: int
        Parameter-initialization and measurement-batching seed.
    Returns:
    --------
    TrainingResult
        Fitted parameters, final full-data loss, and sampled training history.
    """
    number_of_measurements = problem.number_of_measurements
    batch_size = config.batch_size or number_of_measurements
    if batch_size > number_of_measurements:
        raise ValueError(
            f"`batch_size` is {batch_size}, but the forward model has only "
            f"{number_of_measurements} measurements."
        )

    initialization_key = jax.random.PRNGKey(seed)
    batch_key = jax.random.fold_in(initialization_key, 1)
    parameters = model.init(initialization_key, coordinates)["params"]

    optimizer = optax.adam(_build_learning_rate_schedule(config))
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=parameters,
        tx=optimizer,
    )

    @jax.jit
    def optimization_step(
        current_state: train_state.TrainState,
        current_batch_key: jax.Array,
    ) -> tuple[train_state.TrainState, jax.Array, jax.Array]:
        updated_batch_key, sampling_key = jax.random.split(current_batch_key)
        if batch_size == number_of_measurements:
            measurement_indices = jnp.arange(number_of_measurements)
        else:
            measurement_indices = jax.random.choice(
                sampling_key,
                number_of_measurements,
                shape=(batch_size,),
                replace=False,
            )

        def training_objective(current_parameters: ParameterTree) -> jax.Array:
            image = current_state.apply_fn(
                {"params": current_parameters},
                coordinates,
            )
            is_full_batch = batch_size == number_of_measurements
            selected_indices = None if is_full_batch else measurement_indices
            return problem.gaussian_negative_log_likelihood(
                jnp.asarray(image).reshape(-1),
                selected_indices,
            )

        loss, gradients = jax.value_and_grad(training_objective)(
            current_state.params
        )
        updated_state = current_state.apply_gradients(grads=gradients)
        return updated_state, loss, updated_batch_key

    recorded_steps: list[int] = []
    loss_history: list[float] = []

    for step in tqdm(
        range(config.number_of_steps),
        desc=f"training seed {seed}",
        disable=not config.show_progress,
    ):
        state, loss, batch_key = optimization_step(
            state,
            batch_key,
        )
        should_record = (
            step % config.log_interval == 0
            or step == config.number_of_steps - 1
        )
        if should_record:
            loss_value = float(loss)
            recorded_steps.append(step)
            loss_history.append(loss_value)

    final_image = state.apply_fn(
        {"params": state.params},
        coordinates,
    )
    final_loss = float(
        problem.gaussian_negative_log_likelihood(
            jnp.asarray(final_image).reshape(-1),
        )
    )
    return TrainingResult(
        state=state,
        final_loss=final_loss,
        recorded_steps=np.asarray(recorded_steps),
        loss_history=np.asarray(loss_history),
    )


def fit_ensemble(
    model: nn.Module,
    coordinates: jax.Array,
    problem: LinearInverseProblem,
    config: TrainingConfig,
    seeds: tuple[int, ...],
) -> list[TrainingResult]:
    """
    Fit independent ensemble members with different random seeds.

    Parameters:
    -----------
    model: nn.Module
        Flax module fitted for every ensemble seed.
    coordinates: jax.Array
        Coordinates used to initialize and render every member.
    problem: LinearInverseProblem
        Shared measurement system.
    config: TrainingConfig
        Base optimization settings. Its seed is replaced for each member.
    seeds: tuple[int, ...]
        Initialization and batching seeds.
    Returns:
    --------
    list[TrainingResult]
        One result for each seed, preserving input order.
    """
    results: list[TrainingResult] = []
    for seed in seeds:
        results.append(
            fit_linear_inverse_problem(
                model,
                coordinates,
                problem,
                config,
                seed=seed,
            )
        )
    return results
