"""Load and validate cached model ensembles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flax.serialization import from_bytes, to_bytes

from .runs import (
    collect_runtime_metadata,
    describe_file,
    json_value,
)

ParameterTree = Any


@dataclass(frozen=True)
class SavedModel:
    """
    Reference a reusable model saved in the model cache.

    Parameters:
    -----------
    directory: Path
        Saved-model directory containing metadata and checkpoints.
    manifest: dict[str, Any]
        Parsed model and training manifest.
    provenance: dict[str, Any]
        Parsed runtime and input provenance.
    """

    directory: Path
    manifest: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def model_id(self) -> str:
        """Unique saved-model identifier."""
        return str(self.manifest["model_id"])

    @property
    def model_name(self) -> str:
        """Human-readable cache name."""
        return str(self.manifest["model_name"])

    @property
    def training_metadata(self) -> dict[str, Any]:
        metadata = self.manifest.get("training")
        if not isinstance(metadata, dict):
            raise ValueError(
                "Saved-model manifest has invalid training metadata."
            )
        return metadata

    @property
    def model_class(self) -> str:
        """Fully qualified class name of the trained model."""
        return str(self.training_metadata["model_class"])

    @property
    def model_metadata(self) -> Any:
        return self.training_metadata["model"]

    @property
    def problem_metadata(self) -> Any:
        return self.training_metadata["problem"]

    @property
    def image_shape(self) -> tuple[int, int]:
        """Image shape used during training."""
        image_shape = self.training_metadata["image_shape"]
        if not isinstance(image_shape, list) or len(image_shape) != 2:
            raise ValueError(
                "Saved-model manifest has an invalid image shape."
            )
        return int(image_shape[0]), int(image_shape[1])

    @property
    def checkpoint_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.directory / member["checkpoint"]
            for member in self.manifest["members"]
        )

    @property
    def member_seeds(self) -> tuple[int, ...]:
        """Return member seeds in checkpoint order."""
        return tuple(
            int(member["seed"]) for member in self.manifest["members"]
        )

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def provenance_path(self) -> Path:
        return self.directory / "provenance.json"

    @property
    def training_history(self) -> tuple[dict[str, Any], ...]:
        payload = _read_json(self.directory / "training_history.json")
        members = payload.get("members")
        if not isinstance(members, list) or not all(
            isinstance(member, dict) for member in members
        ):
            raise ValueError(
                "Saved-model training history must contain object members."
            )
        return tuple(members)

    def load_parameters(
        self,
        template_parameters: ParameterTree,
    ) -> tuple[ParameterTree, ...]:
        """
        Load every parameter tree using a compatible template.

        Parameters:
        -----------
        template_parameters: ParameterTree
            Parameter structure produced by the matching model.

        Returns:
        --------
        tuple[ParameterTree, ...]
            Ensemble parameters in manifest order.
        """
        return tuple(
            from_bytes(
                template_parameters,
                checkpoint_path.read_bytes(),
            )
            for checkpoint_path in self.checkpoint_paths
        )


def open_saved_model(path: str | Path) -> SavedModel:
    """
    Open a saved model and verify all checkpoint digests.

    Parameters:
    -----------
    path: str | Path
        Saved-model directory containing its manifest and provenance.

    Returns:
    --------
    SavedModel
        Parsed and integrity-checked saved-model record.

    Raises:
    -------
    FileNotFoundError
        If saved-model metadata or a checkpoint does not exist.
    ValueError
        If the manifest is invalid or a checkpoint digest differs.
    """
    model_directory = Path(path)
    manifest = _read_json(model_directory / "manifest.json")
    provenance = _read_json(model_directory / "provenance.json")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(
            "Saved-model manifest must contain at least one member."
        )

    saved_model = SavedModel(
        directory=model_directory,
        manifest=manifest,
        provenance=provenance,
    )
    for member, checkpoint_path in zip(
        members,
        saved_model.checkpoint_paths,
        strict=True,
    ):
        checkpoint_description = describe_file(checkpoint_path)
        if (
            checkpoint_description["sha256"]
            != member.get("checkpoint_sha256")
        ):
            raise ValueError(
                f"Checkpoint digest does not match the manifest: {checkpoint_path}"
            )
    return saved_model


def validate_model_inputs(
    saved_model: SavedModel,
    input_paths: dict[str, str | Path],
) -> None:
    """
    Verify current inputs against the training-time digests.

    Parameters:
    -----------
    saved_model: SavedModel
        Saved model whose training inputs are authoritative.
    input_paths: dict[str, str | Path]
        Current files keyed by their training input names.

    Returns:
    --------
    None

    Raises:
    -------
    FileNotFoundError
        If a current input file does not exist.
    ValueError
        If input names or digests differ from training provenance.
    """
    recorded_inputs = saved_model.provenance.get("inputs", {})
    if set(input_paths) != set(recorded_inputs):
        raise ValueError(
            "Current problem inputs do not match the saved model input set."
        )
    for input_name, input_path in input_paths.items():
        current_description = describe_file(input_path)
        if (
            current_description["sha256"]
            != recorded_inputs[input_name]["sha256"]
        ):
            raise ValueError(
                f"Input {input_name!r} differs from the file used for training."
            )


def _save_model(
    *,
    model_name: str,
    parameters: tuple[ParameterTree, ...],
    seeds: tuple[int, ...],
    final_losses: tuple[float, ...],
    recorded_steps: tuple[np.ndarray, ...],
    loss_histories: tuple[np.ndarray, ...],
    training_metadata: Any,
    input_paths: dict[str, str | Path],
    training_started_at: datetime,
    training_elapsed_seconds: float,
    cache_root: str | Path,
    repository_root: str | Path = Path.cwd(),
) -> SavedModel:
    """Persist a completed training operation in the model cache."""
    model_path = _validate_model_name(model_name)
    histories = _validate_members(
        parameters=parameters,
        seeds=seeds,
        final_losses=final_losses,
        recorded_steps=recorded_steps,
        loss_histories=loss_histories,
    )
    described_inputs = {
        input_name: describe_file(
            input_path,
            relative_to=repository_root,
        )
        for input_name, input_path in input_paths.items()
    }
    runtime_metadata = collect_runtime_metadata(repository_root)
    commit = runtime_metadata["git_commit"]
    commit_short = "nogit" if commit is None else commit[:7]
    model_id = (
        f"{training_started_at.strftime('%Y%m%dT%H%M%SZ')}_{commit_short}"
    )
    model_directory = Path(cache_root) / model_path / model_id
    checkpoint_directory = model_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=False)

    manifest_members = []
    for seed, parameter_tree in zip(
        seeds,
        parameters,
        strict=True,
    ):
        checkpoint_path = checkpoint_directory / f"params_{seed}.msgpack"
        checkpoint_path.write_bytes(to_bytes(parameter_tree))
        checkpoint_description = describe_file(checkpoint_path)
        manifest_members.append(
            {
                "seed": int(seed),
                "checkpoint": str(
                    checkpoint_path.relative_to(model_directory)
                ),
                "checkpoint_sha256": checkpoint_description["sha256"],
                "checkpoint_size_bytes": (
                    checkpoint_description["size_bytes"]
                ),
            }
        )

    _write_json(
        model_directory / "training_history.json",
        {"members": histories},
    )
    manifest = {
        "model_name": model_name,
        "model_id": model_id,
        "training": json_value(training_metadata),
        "training_history": "training_history.json",
        "members": manifest_members,
    }
    provenance = {
        "artifact_type": "saved_model",
        "model_name": model_name,
        "model_id": model_id,
        "training_started_at": training_started_at.isoformat(),
        "training_elapsed_seconds": round(
            training_elapsed_seconds,
            3,
        ),
        **runtime_metadata,
        "inputs": described_inputs,
    }
    _write_json(model_directory / "manifest.json", manifest)
    _write_json(model_directory / "provenance.json", provenance)
    _append_cache_index(
        Path(cache_root),
        model_directory,
        manifest,
        provenance,
    )
    return SavedModel(
        directory=model_directory,
        manifest=manifest,
        provenance=provenance,
    )


def _validate_model_name(model_name: str) -> Path:
    """Validate a semantic, relative model-cache path."""
    model_path = Path(model_name)
    if (
        not model_name
        or model_path.is_absolute()
        or ".." in model_path.parts
        or any(not part for part in model_path.parts)
    ):
        raise ValueError(
            "`model_name` must be a non-empty relative cache path."
        )
    return model_path


def _validate_members(
    *,
    parameters: tuple[ParameterTree, ...],
    seeds: tuple[int, ...],
    final_losses: tuple[float, ...],
    recorded_steps: tuple[np.ndarray, ...],
    loss_histories: tuple[np.ndarray, ...],
) -> list[dict[str, Any]]:
    """Validate aligned member outputs and build readable histories."""
    member_count = len(parameters)
    lengths = {
        member_count,
        len(seeds),
        len(final_losses),
        len(recorded_steps),
        len(loss_histories),
    }
    if member_count == 0 or len(lengths) != 1:
        raise ValueError(
            "Parameters, seeds, losses, and histories must have the same "
            "nonzero length."
        )
    if len(set(seeds)) != member_count:
        raise ValueError("Model seeds must be unique.")

    histories = []
    for seed, final_loss, member_steps, member_losses in zip(
        seeds,
        final_losses,
        recorded_steps,
        loss_histories,
        strict=True,
    ):
        steps = np.asarray(member_steps)
        losses = np.asarray(member_losses)
        if steps.ndim != 1 or losses.ndim != 1 or steps.shape != losses.shape:
            raise ValueError(
                "Each member's recorded steps and losses must be matching "
                "vectors."
            )
        histories.append(
            {
                "seed": int(seed),
                "final_loss": float(final_loss),
                "history": [
                    {"step": int(step), "loss": float(loss)}
                    for step, loss in zip(steps, losses, strict=True)
                ],
            }
        )
    return histories


def _append_cache_index(
    cache_root: Path,
    model_directory: Path,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    """Append a searchable model summary to the cache index."""
    index_path = cache_root / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as index_file:
        index_file.write(
            json.dumps(
                {
                    "model": str(model_directory),
                    "model_name": manifest["model_name"],
                    "model_id": manifest["model_id"],
                    "training_started_at": provenance[
                        "training_started_at"
                    ],
                    "image_shape": manifest["training"]["image_shape"],
                    "model_class": manifest["training"]["model_class"],
                },
                sort_keys=True,
            )
            + "\n"
        )


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    """Write formatted JSON with a trailing newline."""
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
