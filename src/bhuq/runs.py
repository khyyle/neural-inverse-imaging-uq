"""Optional provenance tracking for reproducible research outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            key: json_value(entry)
            for key, entry in asdict(value).items()
        }
    if isinstance(value, dict):
        return {
            str(key): json_value(entry) for key, entry in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_value(entry) for entry in value]
    return value


def describe_file(
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
) -> dict[str, Any]:
    """Return a portable path, byte size, and SHA-256 digest."""
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    recorded_path = file_path
    if relative_to is not None:
        try:
            recorded_path = file_path.relative_to(Path(relative_to).resolve())
        except ValueError:
            pass

    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(recorded_path),
        "size_bytes": file_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def collect_runtime_metadata(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Collect Git and command metadata."""
    resolved_root = Path(repository_root).resolve()
    dirty_output = _git_output(resolved_root, "status", "--porcelain")
    return {
        "git_commit": _git_output(resolved_root, "rev-parse", "HEAD"),
        "git_dirty": bool(dirty_output),
        "command": sys.argv,
    }


class ExperimentRun:
    """
    Persist a coherent research result with provenance.

    Constructing an instance only validates configuration. Entering the
    context creates `results_root/name/<timestamp>_<git-sha>/`, writes the
    resolved configuration, and starts elapsed-time and failure tracking.

    While the context is active, callers may record input files, metrics,
    numerical arrays, model evaluations, and any number of figures. Every
    artifact receives a digest and an entry in `artifacts.json`. Saving a model
    evaluation also records hashes of its saved-model manifest and training
    provenance.

    Leaving the context writes terminal metrics and provenance, appends the
    run to `results_root/index.jsonl`, and records exceptions as failed runs.

    Parameters:
    -----------
    name: str
        Stable result grouping beneath `results_root`.
    config: Any
        Dataclass describing the research question and analysis settings.
    results_root: Path
        Parent directory for tracked runs.
    repository_root: Path
        Git repository used for provenance.

    Examples:
    ---------
    Save one evaluation and several figures from the same in-memory results:

    ```
    with ExperimentRun("uq_comparison", config) as run:
        run.save_evaluation("sgr_a", evaluation)
        run.save_figure("uncertainty_maps", uncertainty_figure)
        run.save_figure("sparsification", sparsification_figure)
    ```
    """

    def __init__(
        self,
        name: str,
        config: Any,
        *,
        results_root: str | Path = Path("results"),
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize run state without touching the filesystem."""
        if not name or any(character.isspace() for character in name):
            raise ValueError(
                "`name` must be non-empty and contain no whitespace."
            )
        if not is_dataclass(config) or isinstance(config, type):
            raise TypeError("`config` must be a dataclass instance.")

        self.name = name
        self.config = config
        self.results_root = Path(results_root)
        self.repository_root = Path(
            repository_root or Path.cwd()
        ).resolve()
        self.metrics: dict[str, Any] = {}
        self.inputs: dict[str, dict[str, Any]] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.saved_models: dict[str, dict[str, Any]] = {}
        self.run_directory: Path | None = None
        self.artifact_directory: Path | None = None
        self._started_at: datetime | None = None
        self._start_time: float | None = None
        self._is_finished = False

    def record_input(self, name: str, path: str | Path) -> None:
        """
        Record an input path, size, and digest.

        Parameters:
        -----------
        name: str
            Stable input label.
        path: str | Path
            Existing input file.

        Returns:
        --------
        None
        """
        self._require_started()
        self.inputs[name] = describe_file(
            path,
            relative_to=self.repository_root,
        )
        self._write_provenance(status="running")

    def log_metric(self, name: str, value: Any) -> None:
        """
        Add or replace a named JSON-compatible metric.

        Parameters:
        -----------
        name: str
            Metric name, optionally using slash-separated hierarchy.
        value: Any
            Scalar or JSON-compatible structured value.

        Returns:
        --------
        None
        """
        self._require_started()
        self.metrics[name] = json_value(value)

    def save_arrays(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        **arrays: np.ndarray,
    ) -> Path:
        """
        Save named numerical arrays and register their provenance.

        Parameters:
        -----------
        name: str
            Semantic artifact name without an extension.
        metadata: dict[str, Any] | None
            Optional description of the artifact's derivation.
        **arrays: np.ndarray
            Named arrays stored in one compressed NPZ file.

        Returns:
        --------
        Path
            Saved NPZ artifact path.
        """
        _, artifact_directory = self._require_started()
        artifact_path = artifact_directory / f"{name}.npz"
        np.savez_compressed(artifact_path, **arrays)
        self._record_artifact(
            name,
            artifact_path,
            kind="arrays",
            metadata=metadata,
        )
        return artifact_path

    def save_figure(
        self,
        name: str,
        figure: Any,
        *,
        metadata: dict[str, Any] | None = None,
        dpi: int = 115,
    ) -> Path:
        """
        Save a figure and register its provenance.

        Parameters:
        -----------
        name: str
            Semantic artifact name without an extension.
        figure: Any
            Matplotlib-compatible object exposing `savefig`.
        metadata: dict[str, Any] | None
            Optional description of the figure's derivation.
        dpi: int
            PNG resolution.

        Returns:
        --------
        Path
            Saved PNG artifact path.
        """
        _, artifact_directory = self._require_started()
        artifact_path = artifact_directory / f"{name}.png"
        figure.savefig(
            artifact_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        self._record_artifact(
            name,
            artifact_path,
            kind="figure",
            metadata=metadata,
        )
        return artifact_path

    def save_evaluation(self, name: str, evaluation: Any) -> Path:
        """
        Save one model evaluation with model and method references.

        Parameters:
        -----------
        name: str
            Semantic label used to prefix arrays and metrics.
        evaluation: ModelEvaluationResult
            In-memory model evaluation to persist.

        Returns:
        --------
        Path
            Saved NPZ uncertainty artifact path.

        Raises:
        -------
        ValueError
            If the evaluation does not reference a saved model.
        """
        if evaluation.saved_model is None:
            raise ValueError(
                "Persisted model evaluations require a saved model."
            )
        self._record_saved_model(name, evaluation.saved_model)
        for metric_name, metric_value in evaluation.metrics.items():
            self.log_metric(f"{name}/{metric_name}", metric_value)
        return self.save_arrays(
            f"{name}_uncertainty",
            metadata={
                "saved_model_reference": name,
                "methods": evaluation.config.methods,
            },
            **evaluation.arrays,
        )

    def _finish(self, *, status: str = "finished") -> None:
        """Finalize metrics, artifacts, provenance, and the run index."""
        if self._is_finished:
            return
        run_directory, _ = self._require_started()
        self._write_json("metrics.json", self.metrics)
        self._write_json("artifacts.json", self.artifacts)
        self._write_provenance(status=status)

        index_path = self.results_root / "index.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("a", encoding="utf-8") as index_file:
            index_file.write(
                json.dumps(
                    {
                        "run": str(run_directory),
                        "experiment": self.name,
                        "status": status,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        self._is_finished = True

    def __enter__(self) -> ExperimentRun:
        """
        Create and initialize the tracked run directory.

        Returns:
        --------
        ExperimentRun
            Active run used inside the context manager.
        """
        if self.run_directory is not None:
            raise RuntimeError("ExperimentRun cannot be entered twice.")
        self._started_at = datetime.now(UTC)
        self._start_time = time.monotonic()
        commit_short = (
            _git_output(
                self.repository_root,
                "rev-parse",
                "--short=7",
                "HEAD",
            )
            or "nogit"
        )
        timestamp = self._started_at.strftime("%Y%m%dT%H%M%SZ")
        self.run_directory = (
            self.results_root / self.name / f"{timestamp}_{commit_short}"
        )
        self.artifact_directory = self.run_directory / "artifacts"
        self.artifact_directory.mkdir(parents=True, exist_ok=False)
        self._write_json("config.json", self.config)
        self._write_json("artifacts.json", self.artifacts)
        self._write_provenance(status="running")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> bool:
        """
        Finalize the run while allowing exceptions to propagate.

        Parameters:
        -----------
        exception_type: type[BaseException] | None
            Exception class raised inside the context, if any.
        exception: BaseException | None
            Exception instance raised inside the context, if any.
        traceback: Any
            Associated traceback, if any.

        Returns:
        --------
        bool
            Always `False`, so exceptions propagate to the caller.
        """
        if exception_type is None:
            self._finish()
        else:
            self._finish(status=f"failed:{exception_type.__name__}")
        return False

    def _record_artifact(
        self,
        name: str,
        path: Path,
        *,
        kind: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Register one saved artifact and immediately update its manifest."""
        run_directory, _ = self._require_started()
        self.artifacts[name] = {
            "kind": kind,
            "path": str(path.relative_to(run_directory)),
            "sha256": describe_file(path)["sha256"],
            "metadata": json_value(metadata or {}),
        }
        self._write_json("artifacts.json", self.artifacts)

    def _write_provenance(self, *, status: str) -> None:
        """Write current run status and environment metadata."""
        if self._started_at is None or self._start_time is None:
            raise RuntimeError("ExperimentRun has not been entered.")
        self._write_json(
            "provenance.json",
            {
                "experiment": self.name,
                "status": status,
                **collect_runtime_metadata(self.repository_root),
                "started_at": self._started_at.isoformat(),
                "elapsed_seconds": round(
                    time.monotonic() - self._start_time,
                    3,
                ),
                "inputs": self.inputs,
                "saved_models": self.saved_models,
            },
        )

    def _record_saved_model(self, name: str, saved_model: Any) -> None:
        """Record one immutable saved-model reference."""
        self._require_started()
        self.saved_models[name] = {
            "model_name": saved_model.model_name,
            "model_id": saved_model.model_id,
            "manifest_sha256": describe_file(
                saved_model.manifest_path
            )["sha256"],
            "provenance_sha256": describe_file(
                saved_model.provenance_path
            )["sha256"],
        }
        self._write_provenance(status="running")

    def _write_json(self, name: str, payload: Any) -> None:
        """Write JSON inside the active run directory."""
        run_directory, _ = self._require_started()
        (run_directory / name).write_text(
            json.dumps(
                json_value(payload),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _require_started(self) -> tuple[Path, Path]:
        """Return active directories or fail before any output operation."""
        if self.run_directory is None or self.artifact_directory is None:
            raise RuntimeError(
                "ExperimentRun must be started before recording outputs."
            )
        if self._is_finished:
            raise RuntimeError("ExperimentRun has already finished.")
        return self.run_directory, self.artifact_directory


def _git_output(repository_root: Path, *arguments: str) -> str | None:
    """Run a read-only Git query and return stripped output when available."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
