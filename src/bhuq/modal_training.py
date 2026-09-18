"""Run independent model-training seeds on Modal GPUs."""

from __future__ import annotations

import importlib
import logging
import tempfile
import time
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, cast

import jax
import modal
import numpy as np
from flax import linen as nn
from flax.serialization import from_bytes, to_bytes

from .forward import LinearInverseProblem
from .model_training import (
    TrainedModel,
    TrainedModelMember,
    save_trained_model,
    train_model,
)
from .models import CoordinateConvention
from .training import TrainingConfig

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTE_SOURCE_ROOT = Path("/root/src")
REMOTE_SCRIPT_ROOT = Path("/root/scripts")
DEFAULT_MODAL_GPU = "A10G"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
DEFAULT_MAX_CONTAINERS = 5

TrainingBuilder = Callable[
    [Any, Any],
    tuple[nn.Module, jax.Array, LinearInverseProblem],
]


@dataclass(frozen=True)
class ModalExecutionConfig:
    """
    Configure Modal resources independently from model optimization.

    Parameters:
    -----------
    gpu: str
        Modal GPU resource name, such as `A10G`, assigned to each member.
    max_parallel: int | None
        Maximum members trained concurrently. `None` permits every requested
        seed to run concurrently.
    timeout_seconds: int
        Maximum runtime of one ensemble member.
    """

    gpu: str = DEFAULT_MODAL_GPU
    max_parallel: int | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        """Reject invalid resource requests before starting a Modal app."""
        if not self.gpu.strip():
            raise ValueError("`gpu` must be a non-empty Modal GPU name.")
        if self.max_parallel is not None and self.max_parallel <= 0:
            raise ValueError("`max_parallel` must be positive or `None`.")
        if self.timeout_seconds <= 0:
            raise ValueError("`timeout_seconds` must be positive.")


@dataclass(frozen=True)
class ModalTrainingRequest:
    """
    Describe a domain-owned training recipe sent to Modal workers.

    The builder belongs to the CT or VLBI training script. Modal only stages
    the problem's declared input files, invokes that builder, and trains seeds.

    Parameters:
    -----------
    builder_path: str
        Import path formatted as `module:function`. The function must accept
        problem and model configuration objects and return the model,
        coordinates, and inverse problem.
    problem_config: Any
        Domain-specific problem configuration.
    model_config: Any
        Domain-specific model configuration.
    training_config: TrainingConfig
        Provider-independent optimization settings.
    coordinate_convention: CoordinateConvention
        Ordering and endpoints represented by the model coordinates.
    model_name: str
        Semantic model-cache path used after members return locally.
    staged_inputs: tuple[tuple[str, str, bytes], ...]
        Problem-config field, original filename, and contents for each input.
    """

    builder_path: str
    problem_config: Any
    model_config: Any
    training_config: TrainingConfig
    coordinate_convention: CoordinateConvention
    model_name: str
    staged_inputs: tuple[tuple[str, str, bytes], ...]

    @classmethod
    def from_configs(
        cls,
        *,
        builder_path: str,
        problem_config: Any,
        model_config: Any,
        training_config: TrainingConfig,
        coordinate_convention: CoordinateConvention,
        model_name: str,
    ) -> ModalTrainingRequest:
        """Create a request and capture every declared problem input."""
        _validate_builder_path(builder_path)
        if not is_dataclass(problem_config) or isinstance(problem_config, type):
            raise TypeError("`problem_config` must be a dataclass instance.")

        input_paths = getattr(problem_config, "input_paths", None)
        if not isinstance(input_paths, dict):
            raise TypeError("`problem_config.input_paths` must be a dictionary.")
        declared_paths = {Path(path).resolve() for path in input_paths.values()}
        path_fields = {
            field.name: value
            for field in fields(problem_config)
            if isinstance(
                value := getattr(problem_config, field.name),
                Path,
            )
        }
        matching_fields = {
            field_name: path
            for field_name, path in path_fields.items()
            if path.resolve() in declared_paths
        }
        matched_paths = {path.resolve() for path in matching_fields.values()}
        if matched_paths != declared_paths:
            raise ValueError(
                "Every declared input path must correspond to a Path field on `problem_config`."
            )

        staged_inputs = tuple(
            (field_name, path.name, path.read_bytes())
            for field_name, path in matching_fields.items()
        )
        return cls(
            builder_path=builder_path,
            problem_config=problem_config,
            model_config=model_config,
            training_config=training_config,
            coordinate_convention=coordinate_convention,
            model_name=model_name,
            staged_inputs=staged_inputs,
        )


modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(PROJECT_ROOT))
    .uv_pip_install("jax[cuda12]==0.11.1")
    .env({"PYTHONPATH": (f"{REMOTE_SOURCE_ROOT}:{REMOTE_SCRIPT_ROOT}")})
    .add_local_dir(PROJECT_ROOT / "src", remote_path=REMOTE_SOURCE_ROOT)
    .add_local_dir(PROJECT_ROOT / "scripts", remote_path=REMOTE_SCRIPT_ROOT)
)
app = modal.App("bhuq-ensemble-training")


@app.function(
    image=modal_image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    max_containers=DEFAULT_MAX_CONTAINERS,
)
def _train_member(
    request: ModalTrainingRequest,
    seed: int,
) -> dict[str, Any]:
    """Train one member and return host-resident parameters and diagnostics."""
    with tempfile.TemporaryDirectory() as directory:
        remote_problem_config = _stage_problem_config(
            request,
            Path(directory),
        )
        builder = _load_training_builder(request.builder_path)
        model, coordinates, problem = builder(
            remote_problem_config,
            request.model_config,
        )
        trained_model = train_model(
            model,
            coordinates,
            problem,
            request.training_config,
            (seed,),
            coordinate_convention=request.coordinate_convention,
            save=False,
        )

    member = trained_model.members[0]
    host_parameters = jax.tree.map(
        lambda value: np.asarray(jax.device_get(value)),
        member.parameters,
    )
    return {
        "seed": member.seed,
        "parameters": to_bytes(host_parameters),
        "final_loss": member.final_loss,
        "recorded_steps": member.recorded_steps.tolist(),
        "loss_history": member.loss_history.tolist(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }


def train_ensemble_on_modal(
    request: ModalTrainingRequest,
    seeds: tuple[int, ...],
    *,
    execution: ModalExecutionConfig | None = None,
    cache_root: str | Path = Path("model_cache"),
) -> TrainedModel:
    """
    Train independent seeds on Modal and save one local model bundle.

    The model and inverse problem are reconstructed by the domain builder in
    each remote container. Returned members are ordered by the requested seeds,
    reconstructed locally, and saved through the standard model-cache API.
    """
    resolved_seeds = _validate_seeds(seeds)
    resolved_execution = execution or ModalExecutionConfig()
    max_containers = min(
        resolved_execution.max_parallel or len(resolved_seeds),
        len(resolved_seeds),
    )
    remote_function = _train_member.with_options(
        gpu=resolved_execution.gpu,
        max_containers=max_containers,
        timeout=resolved_execution.timeout_seconds,
    )

    training_started_at = datetime.now(UTC)
    start_time = time.monotonic()
    with modal.enable_output():
        with app.run():
            calls = [remote_function.spawn(request, seed) for seed in resolved_seeds]
            remote_members = list(modal.FunctionCall.gather(*calls))
    elapsed_seconds = time.monotonic() - start_time

    for member in remote_members:
        LOGGER.info(
            "Modal member seed %s used JAX backend %s; devices: %s",
            member["seed"],
            member["backend"],
            member["devices"],
        )

    builder = _load_training_builder(request.builder_path)
    model, coordinates, problem = builder(
        request.problem_config,
        request.model_config,
    )
    trained_model = _restore_trained_model(
        model,
        coordinates,
        problem,
        request.coordinate_convention,
        resolved_seeds,
        remote_members,
        training_started_at,
        elapsed_seconds,
    )
    return save_trained_model(
        trained_model,
        model_name=request.model_name,
        training_config=request.training_config,
        model_metadata=request.model_config,
        problem_metadata=request.problem_config,
        cache_root=cache_root,
    )


def _stage_problem_config(
    request: ModalTrainingRequest,
    input_directory: Path,
) -> Any:
    """Replace local problem paths with temporary container paths."""
    replacements = {}
    for field_name, filename, contents in request.staged_inputs:
        remote_path = input_directory / field_name / Path(filename).name
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        remote_path.write_bytes(contents)
        replacements[field_name] = remote_path
    return replace(request.problem_config, **replacements)


def _load_training_builder(builder_path: str) -> TrainingBuilder:
    """Import a domain training builder from its stable module path."""
    module_name, _, function_name = builder_path.partition(":")
    module = importlib.import_module(module_name)
    builder = getattr(module, function_name, None)
    if not callable(builder):
        raise TypeError(f"Training builder is not callable: {builder_path}")
    return cast(TrainingBuilder, builder)


def _restore_trained_model(
    model: nn.Module,
    coordinates: jax.Array,
    problem: LinearInverseProblem,
    coordinate_convention: CoordinateConvention,
    seeds: tuple[int, ...],
    remote_members: list[dict[str, Any]],
    training_started_at: datetime,
    elapsed_seconds: float,
) -> TrainedModel:
    """Restore remote member payloads as the standard in-memory model."""
    members_by_seed = {int(member["seed"]): member for member in remote_members}
    if len(members_by_seed) != len(remote_members):
        raise ValueError("Modal returned duplicate ensemble seeds.")
    if set(members_by_seed) != set(seeds):
        raise ValueError("Modal returned different seeds than were requested.")

    template_parameters = model.init(
        jax.random.PRNGKey(0),
        coordinates,
    )["params"]
    members = tuple(
        TrainedModelMember(
            seed=seed,
            parameters=from_bytes(
                template_parameters,
                members_by_seed[seed]["parameters"],
            ),
            final_loss=float(members_by_seed[seed]["final_loss"]),
            recorded_steps=np.asarray(members_by_seed[seed]["recorded_steps"]),
            loss_history=np.asarray(members_by_seed[seed]["loss_history"]),
        )
        for seed in seeds
    )
    return TrainedModel(
        model=model,
        coordinates=coordinates,
        coordinate_convention=coordinate_convention,
        members=members,
        image_shape=problem.image_shape,
        training_started_at=training_started_at,
        training_elapsed_seconds=elapsed_seconds,
        saved_model=None,
    )


def _validate_builder_path(builder_path: str) -> None:
    """Require one unambiguous importable function path."""
    module_name, separator, function_name = builder_path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("`builder_path` must be formatted as `module:function`.")


def _validate_seeds(seeds: tuple[int, ...]) -> tuple[int, ...]:
    """Require at least one unique ensemble seed."""
    resolved_seeds = tuple(seeds)
    if not resolved_seeds:
        raise ValueError("`seeds` must contain at least one value.")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("`seeds` must not contain duplicates.")
    return resolved_seeds
