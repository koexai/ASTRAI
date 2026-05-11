"""
Module for loading and saving model checkpoints and preprocessing artifacts.
Provides functions to load characterizer and generator models along with their
associated scalers and PCA, as well as saving checkpoints after training.
"""

import os
import shutil
import joblib
import yaml
import torch
import numpy as np
import pandas as pd

from models.split_mlp import SplitMLPRegressor
from models.residual_blocks import MLPWithResiduals


def load_config(path):
    """Load YAML configuration from disk."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_scalers_and_pca(ckpt, exp_dir):
    """Load scalers and PCA from disk given checkpoint info.
    ckpt: dict with keys "x_scaler", "y_scaler", "pca" containing filenames
    exp_dir: directory where the checkpoint files are located
    Returns:
    x_scaler: the loaded StandardScaler for input curves
    y_scaler: the loaded StandardScaler for target parameters
    pca: the loaded PCA transformer
    """
    x_scaler = joblib.load(os.path.join(exp_dir, ckpt["x_scaler"]))
    y_scaler = joblib.load(os.path.join(exp_dir, ckpt["y_scaler"]))
    pca = joblib.load(os.path.join(exp_dir, ckpt["pca"]))
    return x_scaler, y_scaler, pca


def load_characterizer(cfg, device, exp_dir):
    """Load characterizer model and preprocessing artifacts from an experiment directory.
    cfg: config dict to determine model architecture and checkpoint names
    device: torch.device to load the model onto
    exp_dir: path to the experiment directory containing checkpoints
    Returns:
    model: the loaded characterizer model
    x_scaler: the loaded StandardScaler for input curves
    y_scaler: the loaded StandardScaler for target parameters
    pca: the loaded PCA transformer
    """
    char_cfg = cfg["characterizer"]
    n_pca = cfg["preprocessing"]["pca_components"]
    n_params = cfg["data"]["n_params"]

    model = SplitMLPRegressor(
        input_dim=n_pca,
        width=char_cfg["model"]["width"],
        num_params=n_params,
        depth=char_cfg["model"]["depth"],
        dropout=char_cfg["model"]["dropout"],
    ).to(device)

    ckpt = char_cfg["checkpoint"]
    model.load_state_dict(
        torch.load(
            os.path.join(exp_dir, ckpt["model"]),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    x_scaler, y_scaler, pca = _load_scalers_and_pca(ckpt, exp_dir)
    return model, x_scaler, y_scaler, pca


def load_generator(cfg, device, exp_dir):
    """Load generator model and preprocessing artifacts from an experiment directory.
    cfg: config dict to determine model architecture and checkpoint names
    device: torch.device to load the model onto
    exp_dir: path to the experiment directory containing checkpoints and preprocessing artifacts
    Returns:
    model: the loaded generator model
    x_scaler: the loaded StandardScaler for input curves
    y_scaler: the loaded StandardScaler for target parameters
    pca: the loaded PCA transformer
    """
    gen_cfg = cfg["generator"]
    n_pca = cfg["preprocessing"]["pca_components"]
    n_params = cfg["data"]["n_params"]

    model = MLPWithResiduals(
        input_dim=n_params,
        width=gen_cfg["model"]["width"],
        out_dim=n_pca,
        depth=gen_cfg["model"]["depth"],
        dropout=gen_cfg["model"]["dropout"],
    ).to(device)

    ckpt = gen_cfg["checkpoint"]
    model.load_state_dict(
        torch.load(
            os.path.join(exp_dir, ckpt["model"]),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    x_scaler, y_scaler, pca = _load_scalers_and_pca(ckpt, exp_dir)
    return model, x_scaler, y_scaler, pca


def load_data(data_path, cfg):
    """Load curves and optional labels from configured data source.

    Supports two formats controlled by ``cfg["data"]["format"]``:

    - ``parquet`` (default): a single Parquet file with curve columns
      ``"0"``..``"n_days-1"`` and named parameter columns.
    - ``npy_csv``: a ``.npy`` file for curves and a ``.csv`` file for
      parameters. When *data_path* is ``None`` the paths are taken from
      the config keys ``curves_path`` and ``params_path``.

    When labels are present, physical parameters are log1p-transformed.

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
    data_cfg = cfg["data"]
    fmt = data_cfg.get("format", "parquet")
    n_days = data_cfg["n_days"]
    param_names = data_cfg["param_names"]

    if fmt == "npy_csv":
        x = np.load(data_cfg["curves_path"]).astype("float32")
        sep = data_cfg.get("params_csv_sep", ",")
        params_df = pd.read_csv(data_cfg["params_path"], sep=sep)
        has_labels = all(p in params_df.columns for p in param_names)
        y = None
        if has_labels:
            y = params_df[param_names].values.astype("float32")
            y = np.log1p(y)
    else:
        curve_cols = [str(i) for i in range(n_days)]
        path = data_path or data_cfg["path"]
        df = pd.read_parquet(path)
        x = df[curve_cols].values.astype("float32")
        has_labels = all(p in df.columns for p in param_names)
        y = None
        if has_labels:
            y = df[param_names].values.astype("float32")
            y = np.log1p(y)

    return x, y


def save_model_checkpoint(
    exp_dir, cfg_checkpoint, model, x_scaler, y_scaler, pca
):
    """Save model state dict and preprocessing artifacts to disk.

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
        os.path.join(exp_dir, cfg_checkpoint["model"]),
    )
    joblib.dump(
        x_scaler,
        os.path.join(exp_dir, cfg_checkpoint["x_scaler"]),
    )
    joblib.dump(
        y_scaler,
        os.path.join(exp_dir, cfg_checkpoint["y_scaler"]),
    )
    joblib.dump(
        pca,
        os.path.join(exp_dir, cfg_checkpoint["pca"]),
    )


def copy_preprocessing_artifacts(prep_dir, exp_dir, cfg_checkpoint):
    """Copy scaler and PCA from preprocessing directory to experiment directory.

    Parameters
    ----------
    prep_dir : str
        Preprocessing directory containing saved scalers and PCA.
    exp_dir : str
        Experiment directory where artifacts are copied.
    cfg_checkpoint : dict
        Configuration dict with keys "x_scaler", "y_scaler", "pca".
    """
    shutil.copy2(
        os.path.join(prep_dir, "x_scaler.pkl"),
        os.path.join(exp_dir, cfg_checkpoint["x_scaler"]),
    )
    shutil.copy2(
        os.path.join(prep_dir, "y_scaler.pkl"),
        os.path.join(exp_dir, cfg_checkpoint["y_scaler"]),
    )
    shutil.copy2(
        os.path.join(prep_dir, "pca.pkl"),
        os.path.join(exp_dir, cfg_checkpoint["pca"]),
    )
