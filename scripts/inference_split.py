"""
inference_split.py - Inference with independently trained characterizer and generator.

Loads separate checkpoints for the two branches and performs:

* **Characterization** -- predict physical parameters from input light curves.
* **Generation** -- reconstruct light curves from ground-truth parameters
  (only when labels are available).
* **Evaluation** -- regression metrics (R2, RMSE, RRMSE, MAE).

Usage::

    python inference_split.py --exp experiments/20260310_120000
    python inference_split.py --exp_char experiments/char --exp_gen experiments/gen
    python inference_split.py --config configs/default_split.yaml --output preds.parquet
"""
import argparse

import numpy as np
import pandas as pd
import torch

from utils.checkpoints import (
    load_config,
    load_characterizer,
    load_generator,
    load_data,
)
from utils.metrics import (
    get_rmse,
    get_mae,
    get_r_squared,
    get_rrmse,
    compute_metrics,
)


def characterize(model, x, x_scaler, y_scaler, pca, device):
    """Curves -> predicted physical parameters.
    x: (n_samples, n_timepoints)
    Returns: (n_samples, n_params)  with inverse scaling applied.
    """
    x_scaled = x_scaler.transform(x)
    x_pca = pca.transform(x_scaled)

    with torch.no_grad():
        pred_sc = model(torch.FloatTensor(x_pca).to(device)).cpu().numpy()

    return y_scaler.inverse_transform(pred_sc)


def generate(model, y, x_scaler, y_scaler, pca, device):
    """Parameters -> predicted light curves.
    y: (n_samples, n_params)
    Returns: (n_samples, n_timepoints) with inverse PCA and scaling applied."""
    y_scaled = y_scaler.transform(y)

    with torch.no_grad():
        pred_pca = model(torch.FloatTensor(y_scaled).to(device)).cpu().numpy()

    return x_scaler.inverse_transform(pca.inverse_transform(pred_pca))


def print_metrics(name, metrics):
    """Pretty-print metrics dict with optional confidence intervals.
    metrics: dict of {metric_name: value} or {metric_name: (mean, std)}"""
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if isinstance(v, tuple):
            print(f"  {k}: {v[0]:.6f} +/- {v[1]:.6f}")
        else:
            print(f"  {k}: {v:.6f}")


def _bootstrap_per_parameter(y, pred_params, param_names, n_boot=100, seed=42):
    """Print per-parameter bootstrap confidence intervals.
    For each parameter, resample the true and predicted values with replacement
    and compute metrics on each resample to get a distribution of metric values.
    """
    rng = np.random.default_rng(seed)
    print(f"\n  Per-parameter metrics (+/- from {n_boot} bootstrap resamples):")
    print(
        f"  {'Parameter':<12} {'R2':>19} {'RMSE':>19} {'RRMSE':>19} {'MAE':>19}"
    )
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
        print(
            f"  {name:<12} {r2_m:.4f}+/-{r2_s:.4f}  {rmse_m:.4f}+/-{rmse_s:.4f}  "
            f"{rrmse_m:.4f}+/-{rrmse_s:.4f}  {mae_m:.4f}+/-{mae_s:.4f}"
        )


def main():
    """Main function to run inference with separate characterizer and generator.
    Loads the characterizer and generator from their respective experiment directories,
    runs characterization and generation, computes metrics, and prints results.
    """
    parser = argparse.ArgumentParser(
        description="Inference with independently trained characterizer and generator."
    )
    parser.add_argument(
        "--config",
        default="configs/default_split.yaml",
        help="Path to split config YAML",
    )
    parser.add_argument(
        "--data", default=None, help="Override data path (parquet format)"
    )
    parser.add_argument(
        "--exp",
        default=None,
        help="Single experiment dir containing both checkpoints",
    )
    parser.add_argument(
        "--exp_char",
        default=None,
        help="Experiment dir for the characterizer checkpoint",
    )
    parser.add_argument(
        "--exp_gen",
        default=None,
        help="Experiment dir for the generator checkpoint",
    )
    parser.add_argument(
        "--output", default=None, help="Path to save predictions as parquet"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    char_dir = args.exp_char or args.exp
    gen_dir = args.exp_gen or args.exp

    n_params = cfg["data"]["n_params"]
    param_names = cfg["data"]["param_names"]

    x, y = load_data(args.data, cfg)
    has_labels = y is not None
    print(f"Loaded {len(x)} samples, labels: {'yes' if has_labels else 'no'}")

    # --- Characterization ---
    pred_params = None
    if char_dir:
        print(f"\nLoading characterizer from: {char_dir}")
        char_model, cx_sc, cy_sc, c_pca = load_characterizer(
            cfg, device, char_dir
        )

        print(f"Running characterization ({len(x)} samples)...")
        pred_params = characterize(char_model, x, cx_sc, cy_sc, c_pca, device)

        if has_labels:
            char_metrics = compute_metrics(y, pred_params, n_cols=n_params)
            print_metrics("CHARACTERIZATION (averaged)", char_metrics)
            _bootstrap_per_parameter(y, pred_params, param_names)
    else:
        print(
            "\nNo characterizer experiment dir provided, skipping characterization."
        )

    # --- Generation ---
    if gen_dir and has_labels:
        print(f"\nLoading generator from: {gen_dir}")
        gen_model, gx_sc, gy_sc, g_pca = load_generator(cfg, device, gen_dir)

        print(f"Running generation ({len(y)} samples)...")
        pred_curves = generate(gen_model, y, gx_sc, gy_sc, g_pca, device)

        gen_metrics = compute_metrics(x, pred_curves)
        print_metrics("GENERATION", gen_metrics)
    elif gen_dir and not has_labels:
        print(
            "\nNo labels available, skipping generation (requires ground-truth params)."
        )
    else:
        print("\nNo generator experiment dir provided, skipping generation.")

    # Save predictions
    if args.output and pred_params is not None:
        results = pd.DataFrame(
            pred_params, columns=[f"pred_{p}" for p in param_names]
        )
        if has_labels:
            for i, p in enumerate(param_names):
                results[f"true_{p}"] = y[:, i]
        results.to_parquet(args.output)
        print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
