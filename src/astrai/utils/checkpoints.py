"""
Module for loading and saving model checkpoints and preprocessing artefacts.
Provides functions to load characterizer and generator models along with their
associated scalers and PCA, as well as saving checkpoints after training.
"""

import os
import hashlib
import yaml
import shutil
from pathlib import Path
import joblib
import torch

from astrai.models.factories import (
    build_characterizer,
    build_generator,
    build_unified_model,
)
from astrai.utils.configuration import load_config
from astrai.utils.data import load_raw_data
from astrai.utils.target_transformations import (
    experiment_target_contract,
    physical_to_transformed,
)


def experiment_artefact_path(exp_dir, configured_path):
    """Resolve a configured artefact name inside one experiment directory.

    Historical configurations may contain a complete experiment path. Only
    the final filename is used when an experiment directory is supplied, so a
    new run cannot write outside its own directory or recreate an old path.
    """
    filename = Path(configured_path).name
    if not filename or filename in {".", ".."}:
        raise ValueError(
            f"Invalid experiment artefact name: {configured_path!r}."
        )
    return Path(exp_dir) / filename


def checkpoint_artefact_paths(exp_dir, cfg_checkpoint):
    """Return all configured checkpoint artefacts keyed by their role."""
    return {
        role: experiment_artefact_path(exp_dir, configured_path)
        for role, configured_path in cfg_checkpoint.items()
        if role in {"model", "x_scaler", "y_scaler", "pca"}
    }


def _checkpoint_path(exp_dir, configured_path):
    """Resolve a checkpoint path with optional experiment containment."""
    if exp_dir is None:
        return Path(configured_path).expanduser()
    return experiment_artefact_path(exp_dir, configured_path)


def _load_scalers_and_pca(ckpt, exp_dir=None):
    """Load scalers and PCA from disk given checkpoint info.
    ckpt: dict with keys "x_scaler", "y_scaler", "pca" containing filenames
    exp_dir: directory where the checkpoint files are located
    Returns:
    x_scaler: the loaded StandardScaler for input curves
    y_scaler: the loaded StandardScaler for target parameters
    pca: the loaded PCA transformer
    """
    x_scaler = joblib.load(
        _checkpoint_path(exp_dir, ckpt["x_scaler"])
    )
    y_scaler = joblib.load(
        _checkpoint_path(exp_dir, ckpt["y_scaler"])
    )
    pca = joblib.load(_checkpoint_path(exp_dir, ckpt["pca"]))
    objects = {"x_scaler": x_scaler, "y_scaler": y_scaler, "pca": pca}
    associations = [getattr(obj, "astrai_association", {}) for obj in objects.values()]
    if any(associations) and len({item.get("bundle_id") for item in associations}) != 1:
        raise ValueError("Checkpoint preprocessing objects belong to different bundles")
    if exp_dir is not None:
        metadata_path = Path(exp_dir) / "metadata.yaml"
        if metadata_path.is_file():
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            if metadata.get("experiment_metadata_version", 0) >= 5:
                from astrai.utils.preprocessing import _learned_state_digest
                selected = metadata.get("checkpoint", {}).get("selected_checkpoint") or {}
                bundle_id = selected.get("preprocessing_bundle_id")
                manifests = [item.get("preprocessing") for item in metadata.get("results", {}).get("folds", [])
                             if item.get("outer_fold") == selected.get("outer_fold")]
                manifest = next((item for item in manifests if item), {})
                if not bundle_id or manifest.get("bundle_id") != bundle_id:
                    raise ValueError("Checkpoint has no verifiable preprocessing association")
                for name, obj in objects.items():
                    if (getattr(obj, "astrai_association", {}).get("bundle_id") != bundle_id
                            or _learned_state_digest(obj) != manifest.get("learned_state", {}).get(name)):
                        raise ValueError(f"Checkpoint preprocessing mismatch: {name}")
                for name in ("model", "x_scaler", "y_scaler", "pca"):
                    path = _checkpoint_path(exp_dir, ckpt[name])
                    record = selected.get("files", {}).get(name, {})
                    if record.get("path") != path.name or record.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise ValueError(f"Checkpoint file differs from its recorded bundle: {name}")
    return x_scaler, y_scaler, pca


def load_characterizer(cfg, device, exp_dir):
    """Load characterizer model and preprocessing artefacts from an experiment directory.
    cfg: config dict to determine model architecture and checkpoint names
    device: torch.device to load the model onto
    exp_dir: path to the experiment directory containing checkpoints
    Returns:
    model: the loaded characterizer model
    x_scaler: the loaded StandardScaler for input curves
    y_scaler: the loaded StandardScaler for target parameters
    pca: the loaded PCA transformer
    """
    experiment_target_contract(exp_dir, cfg)
    char_cfg = cfg["characterizer"]
    model = build_characterizer(cfg).to(device)

    ckpt = char_cfg["checkpoint"]
    model.load_state_dict(
        torch.load(
            _checkpoint_path(exp_dir, ckpt["model"]),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    x_scaler, y_scaler, pca = _load_scalers_and_pca(ckpt, exp_dir)
    return model, x_scaler, y_scaler, pca


def load_generator(cfg, device, exp_dir):
    """Load generator model and preprocessing artefacts from an experiment directory.
    cfg: config dict to determine model architecture and checkpoint names
    device: torch.device to load the model onto
    exp_dir: path to the experiment directory containing checkpoints and preprocessing artefacts
    Returns:
    model: the loaded generator model
    x_scaler: the loaded StandardScaler for input curves
    y_scaler: the loaded StandardScaler for target parameters
    pca: the loaded PCA transformer
    """
    experiment_target_contract(exp_dir, cfg)
    gen_cfg = cfg["generator"]
    model = build_generator(cfg).to(device)

    ckpt = gen_cfg["checkpoint"]
    model.load_state_dict(
        torch.load(
            _checkpoint_path(exp_dir, ckpt["model"]),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    x_scaler, y_scaler, pca = _load_scalers_and_pca(ckpt, exp_dir)
    return model, x_scaler, y_scaler, pca


def load_unified_model(cfg, device, exp_dir=None):
    """Load a unified model and its preprocessing artefacts."""
    experiment_target_contract(exp_dir, cfg)
    model = build_unified_model(cfg).to(device)
    ckpt = cfg["checkpoint"]
    model.load_state_dict(
        torch.load(
            _checkpoint_path(exp_dir, ckpt["model"]),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    x_scaler, y_scaler, pca = _load_scalers_and_pca(ckpt, exp_dir)
    return model, x_scaler, y_scaler, pca


def load_data(data_path, cfg):
    """Load curves and optional transformed labels for legacy callers.

    Supports two formats controlled by ``cfg["data"]["format"]``:

    - ``parquet`` (default): a single Parquet file with curve columns
      ``"0"``..``"n_days-1"`` and named parameter columns.
    - ``npy_csv``: a ``.npy`` file for curves and a ``.csv`` file for
      parameters. When *data_path* is ``None`` the paths are taken from
      the config keys ``curves_path`` and ``params_path``.

    New code should use :func:`astrai.utils.data.load_raw_data` and call the
    target transformation explicitly at the physical/model-space boundary.
    When labels are present, this compatibility adapter applies the configured
    canonical transformation once.

    Parameters
    ----------
    data_path : str or None
        Override path. For ``parquet`` this is the parquet file; for
        ``npy_csv`` it is ignored (paths come from the config).
    cfg : dict
        Parsed YAML configuration (used for column names and n_days).

    Returns
    -------
    tuple
        ``(x, y)`` where *y* is ``None`` when labels are absent.
    """
    x, y_physical = load_raw_data(data_path, cfg)
    y = (
        None
        if y_physical is None
        else physical_to_transformed(y_physical, cfg)
    )
    return x, y


def save_model_checkpoint(
    exp_dir, cfg_checkpoint, model, x_scaler, y_scaler, pca
):
    """Save model state dict and preprocessing artefacts to disk.

    Parameters
    ----------
    exp_dir : str
        Experiment directory where checkpoint files are saved.
    cfg_checkpoint : dict
        Configuration dict with keys "model", "x_scaler", "y_scaler", "pca".
    model : torch.nn.Module
        Trained PyTorch model.
    x_scaler : sklearn.preprocessing.StandardScaler
        Fitted feature scaler.
    y_scaler : sklearn.preprocessing.StandardScaler
        Fitted target scaler.
    pca : sklearn.decomposition.PCA
        Fitted PCA transformer.
    """
    torch.save(
        model.state_dict(),
        experiment_artefact_path(exp_dir, cfg_checkpoint["model"]),
    )
    joblib.dump(
        x_scaler,
        experiment_artefact_path(exp_dir, cfg_checkpoint["x_scaler"]),
    )
    joblib.dump(
        y_scaler,
        experiment_artefact_path(exp_dir, cfg_checkpoint["y_scaler"]),
    )
    joblib.dump(
        pca,
        experiment_artefact_path(exp_dir, cfg_checkpoint["pca"]),
    )


def save_split_checkpoint(exp_dir, cfg_checkpoint, model, prep_dir):
    """Save one split model with its actual fold-specific preprocessing."""
    torch.save(
        model.state_dict(),
        experiment_artefact_path(exp_dir, cfg_checkpoint["model"]),
    )
    copy_preprocessing_artifacts(prep_dir, exp_dir, cfg_checkpoint)
    return checkpoint_artefact_paths(exp_dir, cfg_checkpoint)


def copy_preprocessing_artifacts(prep_dir, exp_dir, cfg_checkpoint):
    """Copy scaler and PCA from preprocessing directory to experiment directory.

    Parameters
    ----------
    prep_dir : str
        Preprocessing directory containing saved scalers and PCA.
    exp_dir : str
        Experiment directory where artefacts are copied.
    cfg_checkpoint : dict
        Configuration dict with keys "x_scaler", "y_scaler", "pca".
    """
    shutil.copy2(
        os.path.join(prep_dir, "x_scaler.pkl"),
        experiment_artefact_path(exp_dir, cfg_checkpoint["x_scaler"]),
    )
    shutil.copy2(
        os.path.join(prep_dir, "y_scaler.pkl"),
        experiment_artefact_path(exp_dir, cfg_checkpoint["y_scaler"]),
    )
    shutil.copy2(
        os.path.join(prep_dir, "pca.pkl"),
        experiment_artefact_path(exp_dir, cfg_checkpoint["pca"]),
    )
