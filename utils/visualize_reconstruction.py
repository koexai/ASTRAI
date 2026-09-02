"""
visualize_reconstruction.py - Visualize the full reconstruction pipeline.

For a single light curve, shows:
1. Original curve
2. Augmented curve (LSST noise + cadence)
3. PCA-only reconstruction (to isolate PCA fidelity)
4. Full model reconstruction (curve -> regressor -> generator -> curve)
5. Residuals between original and reconstructed

Usage::

    python visualize_reconstruction.py --exp experiments/20260306_143000
    python visualize_reconstruction.py --exp experiments/20260306_143000 --index 42
    python visualize_reconstruction.py --exp experiments/20260306_143000 --top 5
"""
import argparse
import os
import numpy as np
import torch
import joblib
import yaml
import matplotlib.pyplot as plt

from models.split_mlp import SplitMLPRegressor, MLPWithResiduals
from models.unified_model import UnifiedModel
from utils.augmentation import apply_lsst_pipeline
from scripts.inference import load_data, load_model


def load_from_experiment(exp_dir, cfg, device):
    """Load model and preprocessing artifacts from an experiment directory.
    exp_dir: path to the experiment directory containing checkpoints and preprocessing artifacts
    cfg: config dict to determine model architecture and checkpoint names
    device: torch.device to load the model onto
    Returns:
    model: the loaded UnifiedModel
    x_scaler: the loaded StandardScaler for input curves
    y_scaler: the loaded StandardScaler for output parameters
    pca: the loaded PCA for input curves
    """
    n_pca = cfg["model"]["pca_components"]
    n_params = cfg["data"]["n_params"]
    width = cfg["model"]["width"]
    depth = cfg["model"]["depth"]
    dropout = cfg["model"]["dropout"]

    regressor = SplitMLPRegressor(
        input_dim=n_pca,
        width=width,
        num_params=n_params,
        depth=depth,
        dropout=dropout,
    )
    generator = MLPWithResiduals(
        input_dim=n_params,
        width=width,
        out_dim=n_pca,
        depth=depth,
        dropout=dropout,
    )
    model = UnifiedModel(regressor, generator).to(device)

    model.load_state_dict(
        torch.load(
            os.path.join(exp_dir, cfg["checkpoint"]["model"]),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    x_scaler = joblib.load(os.path.join(exp_dir, cfg["checkpoint"]["x_scaler"]))
    y_scaler = joblib.load(os.path.join(exp_dir, cfg["checkpoint"]["y_scaler"]))
    pca = joblib.load(os.path.join(exp_dir, cfg["checkpoint"]["pca"]))

    return model, x_scaler, y_scaler, pca


def reconstruct_all(model, x, x_scaler, pca, device):
    """Run the full reconstruction pipeline on all samples.

    Returns
    -------
    model_reconstructed : np.ndarray
        Reconstructed curves, shape (n_samples, n_days).
    pred_params_sc : np.ndarray
        Predicted params in scaled space, shape (n_samples, n_params).
    """
    x_scaled = x_scaler.transform(x)
    x_pca = pca.transform(x_scaled)

    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_pca).to(device)
        pred_params_sc = model.regressor(x_tensor)
        pred_curves_pca = model.generator(pred_params_sc).cpu().numpy()
        pred_params_sc = pred_params_sc.cpu().numpy()

    model_reconstructed = x_scaler.inverse_transform(
        pca.inverse_transform(pred_curves_pca)
    )

    return model_reconstructed, pred_params_sc


def plot_single(
    idx,
    x,
    y,
    model_reconstructed,
    x_scaler,
    pca,
    param_names,
    y_scaler,
    pred_params_sc,
    n_days,
    noise_std,
    per_sample_char_rmse,
    samples_per_day=4,
):
    """Plot the 3-panel visualization for a single sample.
    idx: index of the sample to plot
    x: original curves, shape (n_samples, n_days)
    y: original parameters, shape (n_samples, n_params) or None
    model_reconstructed: reconstructed curves from the model, shape (n_samples, n_days)
    x_scaler: StandardScaler for input curves
    pca: PCA for input curves
    param_names: list of parameter names for printing
    y_scaler: StandardScaler for output parameters (for inverse transforming predictions)
    pred_params_sc: predicted parameters in scaled space, shape (n_samples, n_params)
    n_days: number of days in the curves
    noise_std: standard deviation of Gaussian noise for augmentation
    per_sample_char_rmse: array of characterization RMSE for each sample
    samples_per_day: number of samples to generate per day for the LSST plot
    """

    original_curve = x[idx]
    reconstructed_curve = model_reconstructed[idx]

    # LSST augmentation
    augmented_curves, _ = apply_lsst_pipeline(
        x[idx: idx + 1],
        n_days,
        noise_std,
        samples_per_day=samples_per_day,
    )
    augmented_curve = augmented_curves[0]

    # PCA-only reconstruction
    original_scaled = x_scaler.transform(x[idx : idx + 1])
    original_pca = pca.transform(original_scaled)
    pca_reconstructed = x_scaler.inverse_transform(
        pca.inverse_transform(original_pca)
    )[0]

    # Print parameters
    pred_params = y_scaler.inverse_transform(pred_params_sc[idx : idx + 1])[0]
    pred_params_original = np.expm1(pred_params)

    char_rmse_str = (
        f"Char RMSE: {per_sample_char_rmse[idx]:.6f}"
        if per_sample_char_rmse is not None
        else ""
    )
    print(f"\n--- Sample {idx} ({char_rmse_str}) ---")
    print("Predicted physical parameters:")
    for i, name in enumerate(param_names):
        line = f"  {name}: {pred_params_original[i]:.4f}"
        if y is not None:
            true_params_original = np.expm1(y[idx])
            line += f"  (true: {true_params_original[i]:.4f})"
        print(line)

    # Plot
    time_axis = np.arange(n_days)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    title = f"Sample {idx}"
    if per_sample_char_rmse is not None:
        title += f" — Characterization RMSE: {per_sample_char_rmse[idx]:.6f}"
    fig.suptitle(title, fontsize=13)

    # Compute shared y-axis limits from data with some padding
    all_vals = np.concatenate(
        [
            original_curve,
            augmented_curve,
            pca_reconstructed,
            reconstructed_curve,
        ]
    )
    y_min, y_max = all_vals.min(), all_vals.max()
    y_pad = (y_max - y_min) * 0.1
    ylim = (y_min - y_pad, y_max + y_pad)

    # Panel 1: Original vs Augmented
    axes[0].plot(time_axis, original_curve, label="Original", alpha=0.9)
    axes[0].plot(
        time_axis, augmented_curve, label="Augmented (LSST)", alpha=0.7
    )
    axes[0].set_ylabel("Flux")
    axes[0].set_ylim(*ylim)
    axes[0].set_title("Original vs LSST-Augmented Curve")
    axes[0].legend()

    # Panel 2: Original vs PCA-only vs Model reconstruction
    axes[1].plot(time_axis, original_curve, label="Original", alpha=0.9)
    axes[1].plot(
        time_axis,
        pca_reconstructed,
        label=f"PCA-only ({pca.n_components} comp.)",
        alpha=0.7,
        linestyle="--",
    )
    axes[1].plot(
        time_axis,
        reconstructed_curve,
        label="Model reconstruction",
        alpha=0.7,
        linestyle="-.",
    )
    axes[1].set_ylabel("Flux")
    axes[1].set_ylim(*ylim)
    axes[1].set_title("Reconstruction Comparison")
    axes[1].legend()

    # Panel 3: Residuals
    residual_pca = original_curve - pca_reconstructed
    residual_model = original_curve - reconstructed_curve
    axes[2].plot(
        time_axis, residual_pca, label="Residual (PCA-only)", alpha=0.7
    )
    axes[2].plot(time_axis, residual_model, label="Residual (Model)", alpha=0.7)
    axes[2].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[2].set_xlabel("Time (days)")
    axes[2].set_ylabel("Residual")
    axes[2].set_title("Residuals (Original - Reconstructed)")
    axes[2].legend()

    plt.tight_layout()


def main():
    """Main function to run the visualization.
    Parses command-line arguments, loads the model and data, computes reconstructions,
    and generates plots for selected samples."""
    parser = argparse.ArgumentParser(
        description="Visualize the full reconstruction pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Experiment directory (if not set, loads checkpoints from project root)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index of the sample to visualize",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show the top N samples with lowest reconstruction RMSE",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_days = cfg["data"]["n_days"]
    noise_std = cfg["augmentation"]["noise_std"]
    param_names = cfg["data"]["param_names"]
    samples_per_day = cfg["data"].get("samples_per_day", 4)

    # Load model from experiment dir or project root
    if args.exp:
        model, x_scaler, y_scaler, pca = load_from_experiment(
            args.exp, cfg, device
        )
    else:
        model, x_scaler, y_scaler, pca = load_model(cfg, device)

    # Load data
    x, y = load_data(cfg["data"].get("path"), cfg)

    # Reconstruct all samples
    print("Computing reconstruction for all samples...")
    model_reconstructed, pred_params_sc = reconstruct_all(
        model, x, x_scaler, pca, device
    )

    # Compute per-sample characterization RMSE (predicted vs true params)
    per_sample_char_rmse = None
    if y is not None:
        pred_params = y_scaler.inverse_transform(pred_params_sc)
        per_sample_char_rmse = np.sqrt(np.mean((y - pred_params) ** 2, axis=1))

    # Select which samples to plot
    if args.top is not None:
        if per_sample_char_rmse is None:
            print(
                "ERROR: --top requires ground-truth labels to compute characterization error."
            )
            return
        top_indices = np.argsort(per_sample_char_rmse)[: args.top]
        print(f"\nTop {args.top} samples with lowest characterization RMSE:")
        for rank, idx in enumerate(top_indices, 1):
            print(
                f"  #{rank}: sample {idx}, Char RMSE = {per_sample_char_rmse[idx]:.6f}"
            )
        indices_to_plot = top_indices
    else:
        idx = args.index if args.index is not None else 0
        indices_to_plot = [idx]

    for idx in indices_to_plot:
        plot_single(
            idx,
            x,
            y,
            model_reconstructed,
            x_scaler,
            pca,
            param_names,
            y_scaler,
            pred_params_sc,
            n_days,
            noise_std,
            per_sample_char_rmse,
            samples_per_day=samples_per_day,
        )

    plt.show()


if __name__ == "__main__":
    main()
