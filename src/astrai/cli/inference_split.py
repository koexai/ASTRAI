"""
inference_split.py - Inference with independently trained characterizer and generator.

Loads separate checkpoints for the two branches and performs:

* **Characterization** -- predict physical parameters from input light curves.
* **Generation** -- reconstruct light curves from ground-truth parameters
  (only when labels are available).
* **Evaluation** -- regression metrics (R2, RMSE, RRMSE, MAE).

Usage::

    astrai infer-split --exp experiments/20260310_120000
    astrai infer-split --exp_char experiments/char --exp_gen experiments/gen
    astrai infer-split --config configs/default_split.yaml --output preds.parquet
"""
import argparse

import numpy as np
import pandas as pd
import torch

from astrai.utils.checkpoints import (
    load_config,
    load_characterizer,
    load_generator,
)
from astrai.utils.data import load_raw_data
from astrai.utils.metrics import (
    get_rmse,
    get_mae,
    get_r_squared,
    get_rrmse,
    compute_metrics,
    compute_target_metrics,
)
from astrai.utils.target_transformations import (
    physical_to_scaled,
    physical_to_transformed,
    scaled_to_physical,
    scaled_to_transformed,
    validate_parameter_scalers,
)
from astrai.paths import resolve_config_path


def characterize_scaled(model, x, x_scaler, pca, device):
    """Curves -> predicted parameters in scaled model space."""
    x_scaled = x_scaler.transform(x)
    x_pca = pca.transform(x_scaled)

    with torch.no_grad():
        return model(torch.FloatTensor(x_pca).to(device)).cpu().numpy()


def characterize(model, x, x_scaler, y_scaler, pca, device, cfg=None):
    """Curves -> predicted parameters in physical space."""
    pred_scaled = characterize_scaled(model, x, x_scaler, pca, device)
    return scaled_to_physical(pred_scaled, y_scaler, cfg)


def generate_from_scaled(model, y_scaled, x_scaler, pca, device):
    """Scaled model parameters -> predicted light curves."""
    with torch.no_grad():
        pred_pca = model(torch.FloatTensor(y_scaled).to(device)).cpu().numpy()
    return x_scaler.inverse_transform(pca.inverse_transform(pred_pca))


def generate(model, y, x_scaler, y_scaler, pca, device, cfg=None):
    """Parameters -> predicted light curves.
    y: physical parameters with shape (n_samples, n_params)
    Returns: (n_samples, n_timepoints) with inverse PCA and scaling applied."""
    y_scaled = physical_to_scaled(y, y_scaler, cfg)
    return generate_from_scaled(model, y_scaled, x_scaler, pca, device)


def print_metrics(name, metrics):
    """Pretty-print metrics dict with optional confidence intervals.
    metrics: dict of {metric_name: value} or {metric_name: (mean, std)}"""
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if isinstance(v, tuple):
            print(f"  {k}: {v[0]:.6f} +/- {v[1]:.6f}")
        else:
            print(f"  {k}: {v:.6f}")


def _bootstrap_per_parameter(
    y, pred_params, param_names, space, n_boot=100, seed=42
):
    """Print per-parameter bootstrap confidence intervals.
    For each parameter, resample the true and predicted values with replacement
    and compute metrics on each resample to get a distribution of metric values.
    """
    rng = np.random.default_rng(seed)
    print(
        f"\n  Per-parameter metrics ({space} space; "
        f"+/- from {n_boot} bootstrap resamples):"
    )
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


def main(argv=None):
    """Main function to run inference with separate characterizer and generator.
    Loads the characterizer and generator from their respective experiment directories,
    runs characterization and generation, computes metrics, and prints results.
    """
    parser = argparse.ArgumentParser(
        prog="astrai infer-split",
        description="Inference with independently trained characterizer and generator."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Split config YAML (default: packaged default_split.yaml)",
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
        "--exp-char",
        "--exp_char",
        dest="exp_char",
        default=None,
        help="Experiment dir for the characterizer checkpoint",
    )
    parser.add_argument(
        "--exp-gen",
        "--exp_gen",
        dest="exp_gen",
        default=None,
        help="Experiment dir for the generator checkpoint",
    )
    parser.add_argument(
        "--output", default=None, help="Path to save predictions as parquet"
    )
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config, "default_split.yaml")
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    char_dir = args.exp_char or args.exp
    gen_dir = args.exp_gen or args.exp

    n_params = cfg["data"]["n_params"]
    param_names = cfg["data"]["param_names"]

    x, y_physical = load_raw_data(args.data, cfg)
    has_labels = y_physical is not None
    print(f"Loaded {len(x)} samples, labels: {'yes' if has_labels else 'no'}")

    # --- Characterization ---
    pred_params = None
    if char_dir:
        print(f"\nLoading characterizer from: {char_dir}")
        char_model, cx_sc, cy_sc, c_pca = load_characterizer(
            cfg, device, char_dir
        )

        print(f"Running characterization ({len(x)} samples)...")
        pred_scaled = characterize_scaled(char_model, x, cx_sc, c_pca, device)
        pred_transformed = scaled_to_transformed(pred_scaled, cy_sc)
        pred_params = scaled_to_physical(pred_scaled, cy_sc, cfg)

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
    else:
        print(
            "\nNo characterizer experiment dir provided, skipping characterization."
        )

    # --- Generation ---
    if gen_dir and has_labels:
        print(f"\nLoading generator from: {gen_dir}")
        gen_model, gx_sc, gy_sc, g_pca = load_generator(cfg, device, gen_dir)
        if char_dir:
            validate_parameter_scalers(
                cy_sc,
                gy_sc,
                context="characterizer output and generator input scaling",
            )

        print(f"Running generation ({len(y_physical)} samples)...")
        pred_curves = generate(
            gen_model,
            y_physical,
            gx_sc,
            gy_sc,
            g_pca,
            device,
            cfg,
        )

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
                results[f"true_{p}"] = y_physical[:, i]
        results.to_parquet(args.output)
        print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
