"""Create isolated experiment runs and persist their provenance.

Every training run owns a new directory containing an effective configuration
snapshot, a source archive, lifecycle metadata and generated artefacts. The
metadata is updated atomically so interrupted runs remain distinguishable from
completed ones.
"""

import hashlib
import csv
import os
import shutil
import subprocess
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from astrai.utils.array_dtypes import INDEX_ARRAY_DTYPE, MODEL_ARRAY_DTYPE
from astrai.utils.metrics import get_metric_function
from astrai.utils.runtime_environment import (
    capture_execution_environment,
    capture_runtime_environment,
)
from astrai.utils.target_transformations import (
    target_transform_contract,
    validate_preprocessing_target_contract,
)
from astrai.paths import source_checkout_root, source_snapshot_root


EXPERIMENT_METADATA_VERSION = 4
_CONFIG_SNAPSHOT_NAME = "config.yaml"
_CODE_SNAPSHOT_NAME = "code.zip"
_METADATA_NAME = "metadata.yaml"
_PREPROCESSING_METADATA_SNAPSHOT_NAME = "preprocessing_metadata.yaml"
_SOURCE_CHECKOUT_ROOT = source_checkout_root()
_REPOSITORY_ROOT = _SOURCE_CHECKOUT_ROOT or source_snapshot_root()

_EXCLUDED_CODE_DIRS = {
    "__MACOSX",
    "__pycache__",
    "data",
    "experiments",
    "preprocessed",
    "venv",
}


def _utc_now():
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def _iter_python_files(folder):
    """Yield Python files and their paths relative to the source root."""
    root = Path(folder).resolve()

    for current_dir, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not name.startswith(".")
            and name not in _EXCLUDED_CODE_DIRS
        )

        current_path = Path(current_dir)
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                file_path = current_path / filename
                yield file_path, file_path.relative_to(root).as_posix()


def create_experiment_dir(base_dir="experiments", now=None):
    """Create and return a collision-safe experiment directory.

    Directories use a UTC timestamp with microsecond precision. A numeric
    suffix is added atomically if a caller supplies the same timestamp more
    than once, so an existing run is never reused or overwritten.
    """
    base_path = Path(base_dir).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    timestamp = (now or _utc_now()).strftime("%Y%m%d_%H%M%S_%f")

    for collision_index in range(10_000):
        suffix = "" if collision_index == 0 else f"_{collision_index:02d}"
        destination = base_path / f"{timestamp}{suffix}"
        try:
            destination.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return str(destination)

    raise RuntimeError(
        f"Could not create a unique experiment directory in {base_path}."
    )


def create_pipeline_run_id(now=None):
    """Return a timestamp-based identifier shared by related stage runs."""
    timestamp = (now or _utc_now()).strftime("%Y%m%d_%H%M%S_%f")
    return f"pipeline_{timestamp}_{uuid.uuid4().hex[:8]}"


def save_code(exp_dir, folder=_REPOSITORY_ROOT):
    """Save Python source files recursively into the experiment archive.

    Files retain their paths relative to ``folder``. Generated data,
    experiment outputs, virtual environments, caches, and hidden directories
    are excluded.

    Raises
    ------
    RuntimeError
        If no Python source files are found.
    """
    zip_path = Path(exp_dir) / _CODE_SNAPSHOT_NAME
    source_files = list(_iter_python_files(folder))

    if not source_files:
        raise RuntimeError(
            f"No Python source files found in {Path(folder).resolve()}."
        )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path, archive_path in source_files:
            zipf.write(file_path, archive_path)

    print(f"Code saved to {zip_path}.")


def save_config(
    exp_dir,
    config_path="configs/default.yaml",
    config=None,
):
    """Save the effective configuration as ``config.yaml``.

    Passing ``config`` is preferred because it captures the configuration
    actually used by a programmatic caller. ``config_path`` remains supported
    for callers that only have a source file.
    """
    destination = Path(exp_dir) / _CONFIG_SNAPSHOT_NAME
    if config is None:
        shutil.copy2(Path(config_path).expanduser(), destination)
    else:
        with destination.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Config saved to {destination}.")


def _run_git_command(repository_root, *args):
    """Return stripped Git output, or ``None`` outside a Git worktree."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_metadata(repository_root):
    """Describe the source revision used for an experiment run."""
    if (
        _SOURCE_CHECKOUT_ROOT is None
        and Path(repository_root).resolve() == Path(_REPOSITORY_ROOT).resolve()
    ):
        return {
            "commit": None,
            "branch": None,
            "working_tree_dirty": None,
        }
    commit = _run_git_command(repository_root, "rev-parse", "HEAD")
    if commit is None:
        return {
            "commit": None,
            "branch": None,
            "working_tree_dirty": None,
        }

    branch = _run_git_command(
        repository_root,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
    )
    status = _run_git_command(repository_root, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "working_tree_dirty": None if status is None else bool(status),
    }


def _sha256(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_metadata(value):
    """Convert NumPy and path values into YAML-safe built-in types."""
    if isinstance(value, dict):
        return {
            str(key): _normalise_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def summarise_metric_history(history):
    """Return per-metric values, means and standard deviations."""
    return {
        name: {
            "values": [float(value) for value in values],
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
        }
        for name, values in history.items()
    }


class ExperimentRun:
    """Own an isolated training directory and its lifecycle metadata."""

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.metadata = metadata

    @classmethod
    def start(
        cls,
        *,
        stage,
        config,
        config_path=None,
        exp_dir=None,
        base_dir="experiments",
        preprocessing_dir=None,
        pipeline_run_id=None,
        folds=None,
        base_seed=None,
        device=None,
        checkpoint_metric="R2",
        checkpoint_selection_policy=None,
        repository_root=_REPOSITORY_ROOT,
    ):
        """Create a run, snapshot its inputs and record ``running`` status."""
        target_contract = target_transform_contract(config)
        if exp_dir is None:
            directory = Path(create_experiment_dir(base_dir=base_dir))
        else:
            directory = Path(exp_dir).expanduser().resolve()
            if directory.exists():
                if not directory.is_dir():
                    raise NotADirectoryError(
                        f"Experiment path is not a directory: {directory}."
                    )
                if any(directory.iterdir()):
                    raise FileExistsError(
                        f"Experiment directory is not empty: {directory}. "
                        "Choose a new directory to avoid mixing runs."
                    )
            else:
                directory.mkdir(parents=True, exist_ok=False)

        started_at = _utc_now()
        parameter_names = config.get("data", {}).get("param_names")
        environment = capture_runtime_environment(device=device)
        if checkpoint_selection_policy is None:
            metric_name = checkpoint_metric.rsplit(".", 1)[-1]
            _, metric_mode = get_metric_function(metric_name)
            checkpoint_selection_policy = {
                "dataset": "validation",
                "metric": metric_name,
                "metric_path": checkpoint_metric,
                "mode": metric_mode,
            }

        metadata = {
            "experiment_metadata_version": EXPERIMENT_METADATA_VERSION,
            "run": {
                "id": directory.name,
                "stage": stage,
                "status": "running",
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": None,
                "pipeline_run_id": pipeline_run_id,
            },
            "config": {
                "snapshot": _CONFIG_SNAPSHOT_NAME,
                "source_path": (
                    None
                    if config_path is None
                    else str(Path(config_path).expanduser().resolve())
                ),
            },
            "source": {
                "snapshot": _CODE_SNAPSHOT_NAME,
                "git": _git_metadata(repository_root),
            },
            "data": {
                "n_params": config.get("data", {}).get("n_params"),
                "param_names": parameter_names,
                "target_transform": target_contract,
                "array_dtypes": {
                    "model": MODEL_ARRAY_DTYPE.name,
                    "indices": INDEX_ARRAY_DTYPE.name,
                },
            },
            "execution": {
                "device": None if device is None else str(device),
                "folds": None if folds is None else list(folds),
                "deterministic_algorithms": environment["pytorch"][
                    "deterministic_algorithms"
                ]["enabled"],
            },
            "environment": environment,
            "reproducibility": {
                "base_seed": base_seed,
                "fold_seed_plans": {},
            },
            "preprocessing": {
                "mode": (
                    "in-process"
                    if preprocessing_dir is None
                    else "precomputed"
                ),
                "source_path": (
                    None
                    if preprocessing_dir is None
                    else str(Path(preprocessing_dir).expanduser().resolve())
                ),
                "metadata_snapshot": None,
                "metadata_sha256": None,
                "artefact_schema_version": None,
                "source_run_status": None,
                "legacy_without_metadata": None,
            },
            "results": {
                "dataset_roles": {
                    "training": "parameter optimisation only",
                    "validation": "epoch and outer-fold checkpoint selection",
                    "test": "final performance estimation only",
                },
                "metric_spaces": {
                    "characterization": {
                        "aggregate": "transformed",
                        "per_parameter": ["transformed", "physical"],
                    },
                    "generation": "light_curve",
                },
                "folds": [],
                "summary_dataset": "test",
                "summary": {},
            },
            "checkpoint": {
                "selection_policy": _normalise_metadata(
                    checkpoint_selection_policy
                ),
                "selected_checkpoint": None,
            },
            "artefacts": {},
        }
        run = cls(directory, _normalise_metadata(metadata))
        run._write_metadata()

        try:
            run.metadata["preprocessing"] = (
                cls._snapshot_preprocessing_metadata(
                    directory,
                    preprocessing_dir,
                    config,
                )
            )
            save_config(
                directory,
                config_path=config_path or "configs/default.yaml",
                config=config,
            )
            save_code(directory, folder=repository_root)
        except BaseException as exc:
            run.fail(exc)
            raise

        run._write_metadata()
        return run

    @staticmethod
    def _snapshot_preprocessing_metadata(directory, preprocessing_dir, config):
        """Copy preprocessing metadata when available and describe its source."""
        if preprocessing_dir is None:
            return {
                "mode": "in-process",
                "source_path": None,
                "metadata_snapshot": None,
                "metadata_sha256": None,
                "artefact_schema_version": None,
                "source_run_status": None,
                "legacy_without_metadata": False,
            }

        source_dir = Path(preprocessing_dir).expanduser().resolve()
        source_metadata = source_dir / _METADATA_NAME
        result = {
            "mode": "precomputed",
            "source_path": str(source_dir),
            "metadata_snapshot": None,
            "metadata_sha256": None,
            "artefact_schema_version": None,
            "source_run_status": None,
            "legacy_without_metadata": not source_metadata.is_file(),
        }
        if not source_metadata.is_file():
            return result

        with source_metadata.open(encoding="utf-8") as stream:
            source_record = yaml.safe_load(stream) or {}
        source_status = source_record.get("run", {}).get("status")
        if source_status != "completed":
            raise ValueError(
                "Preprocessing metadata must describe a completed run; "
                f"found status {source_status!r} in {source_metadata}."
            )

        validate_preprocessing_target_contract(source_dir, config)

        snapshot = directory / _PREPROCESSING_METADATA_SNAPSHOT_NAME
        shutil.copy2(source_metadata, snapshot)
        result.update(
            {
                "metadata_snapshot": snapshot.name,
                "metadata_sha256": _sha256(snapshot),
                "artefact_schema_version": source_record.get(
                    "preprocessing_artefact_schema_version"
                ),
                "source_run_status": source_status,
                "legacy_without_metadata": False,
            }
        )
        return result

    def _write_metadata(self):
        """Atomically replace the human-readable metadata file."""
        destination = self.directory / _METADATA_NAME
        temporary = self.directory / f".{_METADATA_NAME}.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                _normalise_metadata(self.metadata),
                stream,
                sort_keys=False,
            )
        os.replace(temporary, destination)

    def save_fold_indices(
        self,
        fold,
        training_indices,
        validation_indices,
        test_indices,
    ):
        """Persist explicit global train, validation and test row indices."""
        fold_directory = self.directory / f"fold_{int(fold)}"
        fold_directory.mkdir(exist_ok=True)
        paths = {}
        for dataset, indices in (
            ("training", training_indices),
            ("validation", validation_indices),
            ("test", test_indices),
        ):
            path = fold_directory / f"{dataset}_indices.npy"
            np.save(path, np.asarray(indices, dtype=INDEX_ARRAY_DTYPE))
            paths[dataset] = path.relative_to(self.directory).as_posix()
        return paths

    def save_training_trace(self, fold, trace):
        """Persist the compact per-epoch selection trace for one fold."""
        fold_directory = self.directory / f"fold_{int(fold)}"
        fold_directory.mkdir(exist_ok=True)
        path = fold_directory / "training_trace.csv"
        fieldnames = (
            "epoch",
            "training_loss",
            "validation_selection_score",
            "learning_rate",
            "selected_checkpoint",
        )
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trace)
        return path.relative_to(self.directory).as_posix()

    def record_fold(
        self,
        fold,
        metrics,
        seed_plan,
        elapsed_seconds,
        *,
        selection=None,
        training=None,
        index_files=None,
        training_trace=None,
    ):
        """Persist one completed outer fold and its selection evidence."""
        fold_key = f"fold_{int(fold)}"
        self.metadata["reproducibility"]["fold_seed_plans"][fold_key] = (
            _normalise_metadata(seed_plan)
        )
        fold_record = {
            "outer_fold": int(fold),
            "elapsed_seconds": float(elapsed_seconds),
            "indices": _normalise_metadata(index_files),
            "checkpoint_selection": _normalise_metadata(selection),
            "training": _normalise_metadata(training),
            "metrics": _normalise_metadata(metrics),
            "training_trace": training_trace,
        }
        self.metadata["results"]["folds"].append(fold_record)
        self._write_metadata()

    def record_execution_environment(self, device=None):
        """Refresh settings configured during the current training process."""
        execution_environment = capture_execution_environment(device=device)
        self.metadata["environment"].update(execution_environment)
        self.metadata["execution"]["deterministic_algorithms"] = (
            execution_environment["pytorch"]["deterministic_algorithms"][
                "enabled"
            ]
        )
        self._write_metadata()

    def record_checkpoint(self, fold, score, files, selected_epoch=None):
        """Record the globally selected validation checkpoint and files."""
        checkpoint_files = {}
        for role, path in files.items():
            artefact_path = Path(path).expanduser().resolve()
            checkpoint_files[role] = {
                "path": artefact_path.relative_to(self.directory).as_posix(),
                "size_bytes": artefact_path.stat().st_size,
                "sha256": _sha256(artefact_path),
            }
        self.metadata["checkpoint"]["selected_checkpoint"] = {
            "outer_fold": int(fold),
            "epoch": None if selected_epoch is None else int(selected_epoch),
            "validation_score": float(score),
            "files": checkpoint_files,
        }
        self._write_metadata()

    def complete(self, summary):
        """Mark the run completed and persist final metrics and a manifest."""
        self.metadata["results"]["summary"] = _normalise_metadata(summary)
        self.metadata["run"].update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now().isoformat(),
            }
        )
        self.metadata["artefacts"] = self._artefact_manifest()
        self._write_metadata()

    def fail(self, exc):
        """Mark the run failed while preserving all partial records."""
        self.metadata["run"].update(
            {
                "status": "failed",
                "completed_at_utc": _utc_now().isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        self.metadata["artefacts"] = self._artefact_manifest()
        self._write_metadata()

    def _artefact_manifest(self):
        """Describe every regular run file except metadata itself."""
        artefacts = {}
        for path in sorted(self.directory.rglob("*")):
            if not path.is_file() or path.name == _METADATA_NAME:
                continue
            relative_path = path.relative_to(self.directory).as_posix()
            artefacts[relative_path] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        return artefacts
