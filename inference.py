"""
inference.py - Run inference with a trained ASTRAI unified model.

Loads a previously trained checkpoint (model weights, scalers, PCA) and
performs:

* **Characterization** -- predict physical parameters from input light curves.
* **Generation** -- reconstruct light curves from ground-truth parameters
  (only when labels are available in the input data).
* **Evaluation** -- compute regression metrics (R2, RMSE, RRMSE, MAE) against
  ground truth when labels are present.

Results can optionally be saved to a Parquet file.

Usage::

    python inference.py                                # default config + data
    python inference.py --data path/to/data.parquet    # custom data
    python inference.py --config configs/custom.yaml   # custom config
    python inference.py --output predictions.parquet   # save predictions
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import joblib
import yaml

from astrai.models import SplitMLPRegressor, MLPWithResiduals, UnifiedModel
from astrai.metrics import get_rmse, get_mae, get_r_squared, get_rrmse


def load_config(path="configs/default.yaml"):
    """Load and parse the YAML training configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Nested configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg, device, exp_dir=None):
    """Reconstruct the UnifiedModel architecture and load trained weights.

    Also loads the fitted StandardScalers (features and targets) and the
    PCA transformer that were persisted alongside the model checkpoint.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    device : torch.device
        Target compute device (CPU / CUDA).
    exp_dir : str, optional
        Experiment directory containing the checkpoint files.  When provided,
        checkpoint filenames from the config are resolved relative to this
        directory.  Otherwise they are loaded from the project root.

    Returns
    -------
    tuple
        ``(model, x_scaler, y_scaler, pca)`` ready for inference.
    """
    n_pca = cfg["model"]["pca_components"]
    n_params = cfg["data"]["n_params"]
    width = cfg["model"]["width"]
    depth = cfg["model"]["depth"]
    dropout = cfg["model"]["dropout"]

    def _path(key):
        name = cfg["checkpoint"][key]
        return os.path.join(exp_dir, name) if exp_dir else name

    regressor = SplitMLPRegressor(input_dim=n_pca, width=width, num_params=n_params, depth=depth, dropout=dropout)
    generator = MLPWithResiduals(input_dim=n_params, width=width, out_dim=n_pca, depth=depth, dropout=dropout)
    model = UnifiedModel(regressor, generator).to(device)

    model.load_state_dict(torch.load(_path("model"), map_location=device, weights_only=True))
    model.eval()

    x_scaler = joblib.load(_path("x_scaler"))
    y_scaler = joblib.load(_path("y_scaler"))
    pca = joblib.load(_path("pca"))

    return model, x_scaler, y_scaler, pca


def load_data(data_path, cfg):
    """Read a dataset and extract light curves and (optional) labels.

    Supports two formats controlled by ``cfg["data"]["format"]``:

    - ``parquet`` (default): a single Parquet file with curve columns
      ``"0"``..``"n_days-1"`` and named parameter columns.
    - ``npy_csv``: a ``.npy`` file for curves and a ``.csv`` file for
      parameters.  When *data_path* is ``None`` the paths are taken from
      the config keys ``curves_path`` and ``params_path``.

    Parameters
    ----------
    data_path : str or None
        Override path.  For ``parquet`` this is the parquet file; for
        ``npy_csv`` it is ignored (paths come from the config).
    cfg : dict
        Parsed YAML configuration (used for column names and n_days).

    Returns
    -------
    tuple
        ``(X, y)`` where *y* is ``None`` when labels are absent.
    """
    data_cfg = cfg["data"]
    fmt = data_cfg.get("format", "parquet")
    n_days = data_cfg["n_days"]
    param_names = data_cfg["param_names"]

    if fmt == "npy_csv":
        X = np.load(data_cfg["curves_path"]).astype("float32")
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
        X = df[curve_cols].values.astype("float32")
        has_labels = all(p in df.columns for p in param_names)
        y = None
        if has_labels:
            y = df[param_names].values.astype("float32")
            y = np.log1p(y)

    return X, y


def characterize(model, X, x_scaler, y_scaler, pca, device):
    """Characterization branch: curves -> predicted physical parameters.

    Applies feature scaling, PCA compression, regressor forward pass,
    and inverse target scaling to return predictions in the original
    (log1p-transformed) parameter space.

    Parameters
    ----------
    model : UnifiedModel
        Trained unified model in eval mode.
    X : numpy.ndarray
        Raw light curves of shape ``(n_samples, n_days)``.
    x_scaler : StandardScaler
        Fitted feature scaler.
    y_scaler : StandardScaler
        Fitted target scaler (for inverse transform).
    pca : PCA
        Fitted PCA transformer.
    device : torch.device
        Compute device.

    Returns
    -------
    numpy.ndarray
        Predicted parameters of shape ``(n_samples, n_params)``.
    """
    X_scaled = x_scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_pca).to(device)
        pred_params_sc = model.regressor(X_tensor).cpu().numpy()

    pred_params = y_scaler.inverse_transform(pred_params_sc)
    return pred_params


def generate(model, y, x_scaler, y_scaler, pca, device):
    """Generation branch: parameters -> predicted light curves.

    Applies target scaling, generator forward pass, inverse PCA, and
    inverse feature scaling to return reconstructed curves in the
    original flux space.

    Parameters
    ----------
    model : UnifiedModel
        Trained unified model in eval mode.
    y : numpy.ndarray
        Ground-truth parameters of shape ``(n_samples, n_params)``.
    x_scaler : StandardScaler
        Fitted feature scaler (for inverse transform).
    y_scaler : StandardScaler
        Fitted target scaler.
    pca : PCA
        Fitted PCA transformer (for inverse transform).
    device : torch.device
        Compute device.

    Returns
    -------
    numpy.ndarray
        Predicted light curves of shape ``(n_samples, n_days)``.
    """
    y_scaled = y_scaler.transform(y)

    with torch.no_grad():
        y_tensor = torch.FloatTensor(y_scaled).to(device)
        pred_curves_pca = model.generator(y_tensor).cpu().numpy()

    pred_curves_scaled = pca.inverse_transform(pred_curves_pca)
    pred_curves = x_scaler.inverse_transform(pred_curves_scaled)
    return pred_curves


def compute_metrics(true, pred, n_cols=None):
    """Compute regression metrics between ground truth and predictions.

    When ``n_cols`` is provided, metrics are computed per-column and then
    averaged (used for characterization with multiple physical parameters).
    Otherwise, arrays are flattened before computing (used for generation).

    Parameters
    ----------
    true : numpy.ndarray
        Ground-truth values.
    pred : numpy.ndarray
        Predicted values (same shape as *true*).
    n_cols : int, optional
        Number of columns for per-column averaging. If ``None``, arrays
        are flattened.

    Returns
    -------
    dict
        Dictionary with keys ``"R2"``, ``"RMSE"``, ``"RRMSE"``, ``"MAE"``.
    """
    if n_cols is not None:
        rmse_vals = [get_rmse(true[:, i], pred[:, i]) for i in range(n_cols)]
        rrmse_vals = [get_rrmse(true[:, i], pred[:, i]) for i in range(n_cols)]
        mae_vals = [get_mae(true[:, i], pred[:, i]) for i in range(n_cols)]
        r2_vals = [get_r_squared(true[:, i], pred[:, i]) for i in range(n_cols)]
        return {
            "R2": (np.mean(r2_vals), np.std(r2_vals)),
            "RMSE": (np.mean(rmse_vals), np.std(rmse_vals)),
            "RRMSE": (np.mean(rrmse_vals), np.std(rrmse_vals)),
            "MAE": (np.mean(mae_vals), np.std(mae_vals)),
        }
    else:
        rmse = get_rmse(true.ravel(), pred.ravel())
        rrmse = get_rrmse(true.ravel(), pred.ravel())
        mae = get_mae(true.ravel(), pred.ravel())
        r2 = get_r_squared(true.ravel(), pred.ravel())
        return {"R2": r2, "RMSE": rmse, "RRMSE": rrmse, "MAE": mae}


def print_metrics(name, metrics):
    """Pretty-print a named block of evaluation metrics.

    Parameters
    ----------
    name : str
        Section label (e.g. "CHARACTERIZATION").
    metrics : dict
        Metric name -> scalar value mapping.
    """
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if isinstance(v, tuple):
            mean, std = v
            print(f"  {k}: {mean:.6f} ± {std:.6f}")
        else:
            print(f"  {k}: {v:.6f}")


def main():
    """Entry point: parse CLI args, load model/data, run inference, and report."""
    parser = argparse.ArgumentParser(description="Run inference with a trained ASTRAI model.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to input parquet file (defaults to config data path)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment directory containing checkpoint files")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save predictions as parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data or cfg["data"].get("path")

    # Load trained checkpoint (model + preprocessing artifacts)
    print(f"Device: {device}")
    if args.exp:
        print(f"Loading model from experiment: {args.exp}")
    else:
        print(f"Loading model from: {cfg['checkpoint']['model']}")
    model, x_scaler, y_scaler, pca = load_model(cfg, device, exp_dir=args.exp)

    # Load input data (labels are optional)
    fmt = cfg["data"].get("format", "parquet")
    if fmt == "npy_csv":
        print(f"Loading data from: {cfg['data']['curves_path']} + {cfg['data']['params_path']}")
    else:
        print(f"Loading data from: {data_path}")
    X, y = load_data(data_path, cfg)
    has_labels = y is not None

    n_params = cfg["data"]["n_params"]
    param_names = cfg["data"]["param_names"]

    # Characterization: curves -> physical parameters
    print(f"\nRunning characterization ({len(X)} samples)...")
    pred_params = characterize(model, X, x_scaler, y_scaler, pca, device)

    if has_labels:
        char_metrics = compute_metrics(y, pred_params, n_cols=n_params)
        print_metrics("CHARACTERIZATION (averaged)", char_metrics)

        # Per-parameter breakdown for detailed diagnostics (with bootstrap ±)
        n_boot = 100
        rng = np.random.default_rng(42)
        print(f"\n  Per-parameter metrics (± from {n_boot} bootstrap resamples):")
        print(f"  {'Parameter':<12} {'R2':>19} {'RMSE':>19} {'RRMSE':>19} {'MAE':>19}")
        print(f"  {'-'*12} {'-'*19} {'-'*19} {'-'*19} {'-'*19}")
        for i, name in enumerate(param_names):
            true_i, pred_i = y[:, i], pred_params[:, i]
            boot = {m: [] for m in ("R2", "RMSE", "RRMSE", "MAE")}
            for _ in range(n_boot):
                idx = rng.integers(0, len(true_i), size=len(true_i))
                boot["R2"].append(get_r_squared(true_i[idx], pred_i[idx]))
                boot["RMSE"].append(get_rmse(true_i[idx], pred_i[idx]))
                boot["RRMSE"].append(get_rrmse(true_i[idx], pred_i[idx]))
                boot["MAE"].append(get_mae(true_i[idx], pred_i[idx]))
            r2_m, r2_s = np.mean(boot["R2"]), np.std(boot["R2"])
            rmse_m, rmse_s = np.mean(boot["RMSE"]), np.std(boot["RMSE"])
            rrmse_m, rrmse_s = np.mean(boot["RRMSE"]), np.std(boot["RRMSE"])
            mae_m, mae_s = np.mean(boot["MAE"]), np.std(boot["MAE"])
            print(f"  {name:<12} {r2_m:.4f}±{r2_s:.4f}  {rmse_m:.4f}±{rmse_s:.4f}  {rrmse_m:.4f}±{rrmse_s:.4f}  {mae_m:.4f}±{mae_s:.4f}")

    # Generation: ground-truth params -> reconstructed curves
    if has_labels:
        print(f"\nRunning generation ({len(y)} samples)...")
        pred_curves = generate(model, y, x_scaler, y_scaler, pca, device)
        gen_metrics = compute_metrics(X, pred_curves)
        print_metrics("GENERATION", gen_metrics)

    # Optionally persist predictions to Parquet
    if args.output:
        results = pd.DataFrame(pred_params, columns=[f"pred_{p}" for p in param_names])
        if has_labels:
            for i, p in enumerate(param_names):
                results[f"true_{p}"] = y[:, i]
        results.to_parquet(args.output)
        print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
