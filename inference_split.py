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
import joblib
import os
import yaml

from astrai.models import SplitMLPRegressor, MLPWithResiduals
from astrai.metrics import get_rmse, get_mae, get_r_squared, get_rrmse


def load_config(path="configs/default_split.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(data_path, cfg):
    """Load curves and optional labels from the configured data source."""
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


def load_characterizer(cfg, device, exp_dir):
    """Load the trained characterizer (SplitMLPRegressor) and its preprocessing."""
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
    model.load_state_dict(torch.load(
        os.path.join(exp_dir, ckpt["model"]), map_location=device, weights_only=True))
    model.eval()

    x_scaler = joblib.load(os.path.join(exp_dir, ckpt["x_scaler"]))
    y_scaler = joblib.load(os.path.join(exp_dir, ckpt["y_scaler"]))
    pca = joblib.load(os.path.join(exp_dir, ckpt["pca"]))

    return model, x_scaler, y_scaler, pca


def load_generator(cfg, device, exp_dir):
    """Load the trained generator (MLPWithResiduals) and its preprocessing."""
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
    model.load_state_dict(torch.load(
        os.path.join(exp_dir, ckpt["model"]), map_location=device, weights_only=True))
    model.eval()

    x_scaler = joblib.load(os.path.join(exp_dir, ckpt["x_scaler"]))
    y_scaler = joblib.load(os.path.join(exp_dir, ckpt["y_scaler"]))
    pca = joblib.load(os.path.join(exp_dir, ckpt["pca"]))

    return model, x_scaler, y_scaler, pca


def characterize(model, X, x_scaler, y_scaler, pca, device):
    """Curves -> predicted physical parameters."""
    X_scaled = x_scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    with torch.no_grad():
        pred_sc = model(torch.FloatTensor(X_pca).to(device)).cpu().numpy()

    return y_scaler.inverse_transform(pred_sc)


def generate(model, y, x_scaler, y_scaler, pca, device):
    """Parameters -> predicted light curves."""
    y_scaled = y_scaler.transform(y)

    with torch.no_grad():
        pred_pca = model(torch.FloatTensor(y_scaled).to(device)).cpu().numpy()

    return x_scaler.inverse_transform(pca.inverse_transform(pred_pca))


def compute_metrics(true, pred, n_cols=None):
    """Compute R2, RMSE, RRMSE, MAE (per-column averaged if n_cols given)."""
    if n_cols is not None:
        rmse = [get_rmse(true[:, i], pred[:, i]) for i in range(n_cols)]
        rrmse = [get_rrmse(true[:, i], pred[:, i]) for i in range(n_cols)]
        mae = [get_mae(true[:, i], pred[:, i]) for i in range(n_cols)]
        r2 = [get_r_squared(true[:, i], pred[:, i]) for i in range(n_cols)]
        return {
            "R2": (np.mean(r2), np.std(r2)),
            "RMSE": (np.mean(rmse), np.std(rmse)),
            "RRMSE": (np.mean(rrmse), np.std(rrmse)),
            "MAE": (np.mean(mae), np.std(mae)),
        }
    else:
        return {
            "R2": get_r_squared(true.ravel(), pred.ravel()),
            "RMSE": get_rmse(true.ravel(), pred.ravel()),
            "RRMSE": get_rrmse(true.ravel(), pred.ravel()),
            "MAE": get_mae(true.ravel(), pred.ravel()),
        }


def print_metrics(name, metrics):
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if isinstance(v, tuple):
            print(f"  {k}: {v[0]:.6f} +/- {v[1]:.6f}")
        else:
            print(f"  {k}: {v:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference with independently trained characterizer and generator.")
    parser.add_argument("--config", default="configs/default_split.yaml",
                        help="Path to split config YAML")
    parser.add_argument("--data", default=None,
                        help="Override data path (parquet format)")
    parser.add_argument("--exp", default=None,
                        help="Single experiment dir containing both checkpoints")
    parser.add_argument("--exp_char", default=None,
                        help="Experiment dir for the characterizer checkpoint")
    parser.add_argument("--exp_gen", default=None,
                        help="Experiment dir for the generator checkpoint")
    parser.add_argument("--output", default=None,
                        help="Path to save predictions as parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve experiment directories
    char_dir = args.exp_char or args.exp
    gen_dir = args.exp_gen or args.exp

    n_params = cfg["data"]["n_params"]
    param_names = cfg["data"]["param_names"]

    # Load data
    X, y = load_data(args.data, cfg)
    has_labels = y is not None
    print(f"Loaded {len(X)} samples, labels: {'yes' if has_labels else 'no'}")

    # --- Characterization ---
    if char_dir:
        print(f"\nLoading characterizer from: {char_dir}")
        char_model, cx_sc, cy_sc, c_pca = load_characterizer(cfg, device, char_dir)

        print(f"Running characterization ({len(X)} samples)...")
        pred_params = characterize(char_model, X, cx_sc, cy_sc, c_pca, device)

        if has_labels:
            char_metrics = compute_metrics(y, pred_params, n_cols=n_params)
            print_metrics("CHARACTERIZATION (averaged)", char_metrics)

            # Per-parameter bootstrap breakdown
            n_boot = 100
            rng = np.random.default_rng(42)
            print(f"\n  Per-parameter metrics (+/- from {n_boot} bootstrap resamples):")
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
                print(f"  {name:<12} {r2_m:.4f}+/-{r2_s:.4f}  {rmse_m:.4f}+/-{rmse_s:.4f}  "
                      f"{rrmse_m:.4f}+/-{rrmse_s:.4f}  {mae_m:.4f}+/-{mae_s:.4f}")
    else:
        pred_params = None
        print("\nNo characterizer experiment dir provided, skipping characterization.")

    # --- Generation ---
    if gen_dir and has_labels:
        print(f"\nLoading generator from: {gen_dir}")
        gen_model, gx_sc, gy_sc, g_pca = load_generator(cfg, device, gen_dir)

        print(f"Running generation ({len(y)} samples)...")
        pred_curves = generate(gen_model, y, gx_sc, gy_sc, g_pca, device)

        gen_metrics = compute_metrics(X, pred_curves)
        print_metrics("GENERATION", gen_metrics)
    elif gen_dir and not has_labels:
        print("\nNo labels available, skipping generation (requires ground-truth params).")
    else:
        pred_curves = None
        print("\nNo generator experiment dir provided, skipping generation.")

    # Save predictions
    if args.output and pred_params is not None:
        results = pd.DataFrame(pred_params, columns=[f"pred_{p}" for p in param_names])
        if has_labels:
            for i, p in enumerate(param_names):
                results[f"true_{p}"] = y[:, i]
        results.to_parquet(args.output)
        print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
