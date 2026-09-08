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

    astrai infer                                # default config + data
    astrai infer --data path/to/data.parquet    # custom data
    astrai infer --config configs/custom.yaml   # custom config
    astrai infer --output predictions.parquet   # save predictions
"""
import argparse
import numpy as np
import pandas as pd
import torch

from astrai.utils.metrics import (
    get_rmse,
    get_mae,
    get_r_squared,
    get_rrmse,
    compute_metrics,
    compute_target_metrics,
)
from astrai.utils.checkpoints import load_unified_model
from astrai.utils.configuration import load_config
from astrai.utils.data import load_raw_data
from astrai.utils.target_transformations import (
    physical_to_scaled,
    physical_to_transformed,
    scaled_to_physical,
    scaled_to_transformed,
)
from astrai.paths import resolve_config_path


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
    return load_unified_model(cfg, device, exp_dir=exp_dir)


def characterize_scaled(model, x, x_scaler, pca, device):
    """Run characterisation and return model outputs in scaled space."""
    x_scaled = x_scaler.transform(x)
    x_pca = pca.transform(x_scaled)

    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_pca).to(device)
        return model.regressor(x_tensor).cpu().numpy()


def characterize_transformed(model, x, x_scaler, y_scaler, pca, device):
    """Run characterisation and return parameters in transformed space."""
    pred_scaled = characterize_scaled(model, x, x_scaler, pca, device)
    return scaled_to_transformed(pred_scaled, y_scaler)


def characterize(model, x, x_scaler, y_scaler, pca, device, cfg=None):
    """Characterization branch: curves -> predicted physical parameters.

    Applies feature scaling, PCA compression, regressor forward pass,
    and the canonical inverse target transformation.

    Parameters
    ----------
    model : UnifiedModel
        Trained unified model in eval mode.
    x : numpy.ndarray
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
    pred_scaled = characterize_scaled(model, x, x_scaler, pca, device)
    return scaled_to_physical(pred_scaled, y_scaler, cfg)


def generate_from_scaled(model, y_scaled, x_scaler, pca, device):
    """Generate light curves from model parameters already in scaled space."""
    with torch.no_grad():
        y_tensor = torch.FloatTensor(y_scaled).to(device)
        pred_curves_pca = model.generator(y_tensor).cpu().numpy()

    pred_curves_scaled = pca.inverse_transform(pred_curves_pca)
    return x_scaler.inverse_transform(pred_curves_scaled)


def generate(model, y, x_scaler, y_scaler, pca, device, cfg=None):
    """Generation branch: parameters -> predicted light curves.

    Applies target scaling, generator forward pass, inverse PCA, and
    inverse feature scaling to return reconstructed curves in the
    original flux space.

    Parameters
    ----------
    model : UnifiedModel
        Trained unified model in eval mode.
    y : numpy.ndarray
        Physical parameters of shape ``(n_samples, n_params)``.
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
    y_scaled = physical_to_scaled(y, y_scaler, cfg)
    return generate_from_scaled(model, y_scaled, x_scaler, pca, device)


def _bootstrap_per_parameter(
    true,
    pred,
    param_names,
    space,
    n_boot=100,
    seed=42,
):
    """Print bootstrap metrics for one explicitly named target space."""
    rng = np.random.default_rng(seed)
    print(
        f"\n  Per-parameter metrics ({space} space; "
        f"± from {n_boot} bootstrap resamples):"
    )
    print(
        f"  {'Parameter':<12} {'R2':>19} {'RMSE':>19} "
        f"{'RRMSE':>19} {'MAE':>19}"
    )
    print(f"  {'-'*12} {'-'*19} {'-'*19} {'-'*19} {'-'*19}")
    for column, name in enumerate(param_names):
        true_i, pred_i = true[:, column], pred[:, column]
        boot = {metric: [] for metric in ("R2", "RMSE", "RRMSE", "MAE")}
        for _ in range(n_boot):
            indices = rng.integers(0, len(true_i), size=len(true_i))
            boot["R2"].append(get_r_squared(true_i[indices], pred_i[indices]))
            boot["RMSE"].append(get_rmse(true_i[indices], pred_i[indices]))
            boot["RRMSE"].append(get_rrmse(true_i[indices], pred_i[indices]))
            boot["MAE"].append(get_mae(true_i[indices], pred_i[indices]))
        formatted = [
            f"{np.mean(boot[metric]):.4f}±{np.std(boot[metric]):.4f}"
            for metric in ("R2", "RMSE", "RRMSE", "MAE")
        ]
        print(f"  {name:<12} {' '.join(formatted)}")


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


def main(argv=None):
    """Entry point: parse CLI args, load model/data, run inference, and report.
    Steps:
    1. Parse command-line args for config, data, experiment, and output path.
    2. Load the YAML config and determine compute device.
    3. Load the trained model and preprocessing artifacts from the experiment directory.
    4. Load the input data (light curves and optionally parameters).
    5. Run characterization to predict parameters from curves
        and compute metrics if labels are available.
    6. Run generation to reconstruct curves from ground-truth parameters
        and compute metrics if labels are available.
    7. Optionally save predictions to a Parquet file."""
    parser = argparse.ArgumentParser(
        prog="astrai infer",
        description="Run inference with a trained ASTRAI model."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to input parquet file (defaults to config data path)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML (default: packaged default.yaml)",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Experiment directory containing checkpoint files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions as parquet",
    )
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config, "default.yaml")
    cfg = load_config(config_path)
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
        print(
            f"Loading data from: {cfg['data']['curves_path']} + {cfg['data']['params_path']}"
        )
    else:
        print(f"Loading data from: {data_path}")
    x, y_physical = load_raw_data(data_path, cfg)
    has_labels = y_physical is not None

    n_params = cfg["data"]["n_params"]
    param_names = cfg["data"]["param_names"]

    # Characterization: curves -> physical parameters
    print(f"\nRunning characterization ({len(x)} samples)...")
    pred_scaled = characterize_scaled(model, x, x_scaler, pca, device)
    pred_transformed = scaled_to_transformed(pred_scaled, y_scaler)
    pred_params = scaled_to_physical(pred_scaled, y_scaler, cfg)

    if has_labels:
        y_transformed = physical_to_transformed(y_physical, cfg)
        char_metrics = compute_target_metrics(
            y_transformed,
            pred_transformed,
            param_names,
            cfg,
        )
        print_metrics(
            "CHARACTERIZATION (transformed-space aggregate)",
            char_metrics["transformed"]["aggregate"],
        )
        _bootstrap_per_parameter(
            y_transformed,
            pred_transformed,
            param_names,
            "transformed",
        )
        _bootstrap_per_parameter(
            y_physical,
            pred_params,
            param_names,
            "physical",
        )

    # Generation: ground-truth params -> reconstructed curves
    if has_labels:
        print(f"\nRunning generation ({len(y_physical)} samples)...")
        pred_curves = generate(
            model,
            y_physical,
            x_scaler,
            y_scaler,
            pca,
            device,
            cfg,
        )
        gen_metrics = compute_metrics(x, pred_curves)
        print_metrics("GENERATION", gen_metrics)

    # Optionally persist predictions to Parquet
    if args.output:
        results = pd.DataFrame(
            pred_params, columns=[f"pred_{p}" for p in param_names]
        )
        if has_labels:
            for i, p in enumerate(param_names):
                results[f"true_{p}"] = y_physical[:, i]
        results.to_parquet(args.output)
        print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
