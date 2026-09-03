"""
preprocess.py - Fit and persist shared preprocessing artifacts (scalers, PCA, augmented data).

Fits PCA and StandardScalers once on the full dataset, then for each K-Fold
split saves the pre-transformed arrays so that training scripts can load them
directly without recomputing.

Usage::

    python preprocess.py
    python preprocess.py --config configs/default_split.yaml
    python preprocess.py --config configs/default_split.yaml --out path/to/run
"""
import argparse
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from utils.augmentation import apply_lsst_pipeline
from utils.checkpoints import load_data
from utils.log_experiments import save_code


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RUNS_DIR = _REPOSITORY_ROOT / "preprocessed"
_CONFIG_SNAPSHOT_NAME = "config.yaml"
_METADATA_NAME = "metadata.yaml"
_ARTEFACT_SCHEMA_VERSION = 1


def _utc_now():
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def _config_run_name(config_path):
    """Return a filesystem-safe run name derived from a config path."""
    if config_path is None:
        return "config"

    name = Path(config_path).stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-_")
    return name or "config"


def _create_run_directory(config_path, out_dir=None, now=None):
    """Create and return a new preprocessing run directory.

    An explicit destination is used exactly as supplied. Otherwise a
    timestamped directory is created below ``preprocessed``. Existing
    destinations are rejected so that prior artefacts cannot be overwritten.
    """
    if out_dir is None:
        timestamp = (now or _utc_now()).strftime("%Y%m%d_%H%M%S")
        destination = _DEFAULT_RUNS_DIR / (
            f"{timestamp}_{_config_run_name(config_path)}"
        )
    else:
        destination = Path(out_dir).expanduser().resolve()

    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Preprocessing output directory already exists: {destination}. "
            "Choose a new --out directory or omit --out to create a "
            "timestamped run."
        ) from exc

    return destination


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
    """Describe the source revision used for a preprocessing run."""
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


def _write_metadata(run_dir, metadata):
    """Write preprocessing run metadata in a human-readable format."""
    metadata_path = Path(run_dir) / _METADATA_NAME
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)


def _save_config_snapshot(run_dir, cfg, config_path=None):
    """Save the exact source config, or serialise a programmatic config."""
    destination = Path(run_dir) / _CONFIG_SNAPSHOT_NAME
    if config_path is not None:
        shutil.copy2(Path(config_path).expanduser(), destination)
        return

    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)


def _initial_metadata(cfg, started_at, repository_root):
    """Build the initial metadata record for a preprocessing run."""
    preprocessing_cfg = cfg["preprocessing"]
    n_splits = preprocessing_cfg["n_splits"]
    return {
        "preprocessing_artefact_schema_version": _ARTEFACT_SCHEMA_VERSION,
        "run": {
            "status": "running",
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": None,
        },
        "config": {
            "snapshot": _CONFIG_SNAPSHOT_NAME,
        },
        "preprocessing": {
            "random_seed": preprocessing_cfg["random_seed"],
            "n_splits": n_splits,
            "folds": list(range(1, n_splits + 1)),
        },
        "git": _git_metadata(repository_root),
    }


def _fit_global_artifacts(x_raw, y_raw, n_pca, out_dir):
    """Fit scalers and PCA on full dataset and save to out_dir.

    Parameters
    ----------
    x_raw : np.ndarray
        Raw input curves, shape (n_samples, n_days).
    y_raw : np.ndarray
        Raw parameters, shape (n_samples, n_params).
    n_pca : int
        Number of PCA components to keep.
    out_dir : str
        Output directory to save fitted artifacts.
    """
    print("Fitting scalers and PCA on full dataset...")
    x_scaler = StandardScaler()
    x_scaler.fit(x_raw)

    y_scaler = StandardScaler()
    y_scaler.fit(y_raw)

    pca = PCA(n_components=n_pca)
    pca.fit(x_scaler.transform(x_raw))
    explained_var = pca.explained_variance_ratio_.sum()
    print(
        f"PCA explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)"
    )

    joblib.dump(x_scaler, os.path.join(out_dir, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(out_dir, "y_scaler.pkl"))
    joblib.dump(pca, os.path.join(out_dir, "pca.pkl"))

    return x_scaler, y_scaler, pca


def _process_fold(
    fold_idx,
    x_raw,
    y_raw,
    train_idx,
    test_idx,
    x_scaler,
    y_scaler,
    pca,
    n_days,
    noise_std,
    samples_per_day,
    out_dir,
):
    """Augment, transform, and save a single fold's data.

    Parameters
    ----------
    fold_idx : int
        Index of the current fold (1-based).
        x_raw : np.ndarray
        Raw input curves, shape (n_samples, n_days).
        y_raw : np.ndarray
        Raw parameters, shape (n_samples, n_params).
        train_idx : np.ndarray
        Indices for training samples in this fold.
        test_idx : np.ndarray
        Indices for test samples in this fold.
        x_scaler : StandardScaler
        Fitted scaler for input curves.
        y_scaler : StandardScaler
        Fitted scaler for parameters.
        pca : PCA
        Fitted PCA for input curves.
        n_days : int
        Number of days in the input curves.
        noise_std : float
        Standard deviation of Gaussian noise for augmentation.
        samples_per_day : int
        Number of augmented samples to generate per day.
        out_dir : str
        Base output directory for this fold's artifacts.
    """
    fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    x_train_clean = x_raw[train_idx]
    x_test_clean = x_raw[test_idx]
    y_train = y_raw[train_idx]
    y_test = y_raw[test_idx]

    np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(fold_dir, "test_idx.npy"), test_idx)

    print("  Applying LSST augmentation...")
    x_train_aug, _ = apply_lsst_pipeline(
        x_train_clean,
        n_days,
        noise_std,
        samples_per_day=samples_per_day,
    )

    x_train_clean_pca = pca.transform(x_scaler.transform(x_train_clean))
    x_train_aug_pca = pca.transform(x_scaler.transform(x_train_aug))
    x_test_pca = pca.transform(x_scaler.transform(x_test_clean))

    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    np.save(os.path.join(fold_dir, "x_train_clean_pca.npy"), x_train_clean_pca)
    np.save(os.path.join(fold_dir, "x_train_aug_pca.npy"), x_train_aug_pca)
    np.save(os.path.join(fold_dir, "x_test_pca.npy"), x_test_pca)
    np.save(os.path.join(fold_dir, "y_train_scaled.npy"), y_train_scaled)
    np.save(os.path.join(fold_dir, "y_test_scaled.npy"), y_test_scaled)
    np.save(os.path.join(fold_dir, "y_test.npy"), y_test)
    np.save(os.path.join(fold_dir, "x_test_clean.npy"), x_test_clean)

    print(f"  Saved to {fold_dir}")


def _generate_preprocessing_artefacts(cfg, out_dir):
    """Generate the numerical artefacts inside an existing run directory."""
    n_days = cfg["data"]["n_days"]
    samples_per_day = cfg["data"].get("samples_per_day", 4)
    noise_std = cfg["augmentation"]["noise_std"]
    n_pca = cfg["preprocessing"]["pca_components"]
    n_splits = cfg["preprocessing"]["n_splits"]
    seed = cfg["preprocessing"]["random_seed"]

    print("Loading data...")
    x_raw, y_raw = load_data(None, cfg)
    y_raw = np.log1p(y_raw)

    x_scaler, y_scaler, pca = _fit_global_artifacts(
        x_raw, y_raw, n_pca, out_dir
    )

    np.save(os.path.join(out_dir, "x_raw.npy"), x_raw)
    np.save(os.path.join(out_dir, "y_raw.npy"), y_raw)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x_raw), 1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")
        _process_fold(
            fold_idx,
            x_raw,
            y_raw,
            train_idx,
            test_idx,
            x_scaler,
            y_scaler,
            pca,
            n_days,
            noise_std,
            samples_per_day,
            out_dir,
        )


def run_preprocessing(cfg, out_dir=None, config_path=None):
    """Run preprocessing in a new, self-contained artefact directory.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    out_dir : str or pathlib.Path or None
        Exact destination for the new run. A timestamped directory below
        ``preprocessed`` is created when omitted.
    config_path : str or pathlib.Path or None
        Source YAML file to preserve verbatim. When omitted, ``cfg`` is
        serialised to the run directory.

    Returns
    -------
    str
        Path to the completed preprocessing run directory.
    """
    if config_path is not None and not Path(config_path).expanduser().is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    started_at = _utc_now()
    metadata = _initial_metadata(cfg, started_at, _REPOSITORY_ROOT)
    run_dir = _create_run_directory(config_path, out_dir, now=started_at)
    _write_metadata(run_dir, metadata)

    try:
        _save_config_snapshot(run_dir, cfg, config_path=config_path)
        save_code(run_dir, folder=_REPOSITORY_ROOT)
        _generate_preprocessing_artefacts(cfg, run_dir)
    except Exception as exc:
        metadata["run"].update(
            {
                "status": "failed",
                "completed_at_utc": _utc_now().isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        _write_metadata(run_dir, metadata)
        raise

    metadata["run"].update(
        {
            "status": "completed",
            "completed_at_utc": _utc_now().isoformat(),
        }
    )
    _write_metadata(run_dir, metadata)

    run_path = str(run_dir)
    print(f"\nPreprocessing complete. Artefacts in: {run_path}")
    print("Use this directory for subsequent training:")
    print(f"  --prep {run_path}")
    return run_path


def main():
    """Main function to run preprocessing.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    out_dir : str
        Output directory for preprocessing artifacts.
    """
    parser = argparse.ArgumentParser(description="ASTRAI shared preprocessing")
    parser.add_argument("--config", default="configs/default_split.yaml")
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Exact output directory for this preprocessing run. If omitted, "
            "a timestamped directory is created under preprocessed/."
        ),
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_preprocessing(
        cfg,
        out_dir=args.out,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
