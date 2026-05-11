"""
plot_results.py - Generate result plots for the ASTRAI split model.

Plot 1: Single LC Reconstruction (best sample by characterization RMSE)
  - Original curve  (blue solid)
  - Desampled points (green dots, LSST cadence on clean curve)
  - Noisy points     (black dots + error bars, LSST cadence + noise)
  - Reconstructed   (red dashed, characterizer -> generator pipeline)

Plot 2: Per-timestep Reconstruction Error on the test set
  - RMSE at each time step across all test samples (log scale)
  - Generator evaluated with teacher-forced true parameters

Usage::

    python plot_results.py \\
        --exp_char experiments/characterizer/YYYYMMDD_HHMMSS \\
        --exp_gen  experiments/generator/YYYYMMDD_HHMMSS

    python plot_results.py \\
        --exp_char experiments/characterizer/YYYYMMDD_HHMMSS \\
        --exp_gen  experiments/generator/YYYYMMDD_HHMMSS \\
        --fold 2 --output_dir plots/
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import lsst
from utils.augmentation import apply_lsst_pipeline
from utils.checkpoints import load_config, load_characterizer, load_generator


def get_lsst_observed_idx(n_days, samples_per_day, seed=42):
    """Return indices of time steps that survive sun + cloud masking."""
    np.random.seed(seed)
    calendar = np.arange(n_days) / samples_per_day
    sun_mask = lsst.sun_masking_np(calendar)
    cloud_mask = lsst.random_cloud_masking(np.ones_like(calendar))
    combined_mask = (1 - sun_mask) * (1 - cloud_mask)
    return np.where(combined_mask == 1)[0]


def generate_curves(y_scaled, gen_model, gen_x_scaler, gen_pca, device):
    """Generator forward pass: scaled params -> reconstructed curves in original space."""
    with torch.no_grad():
        t = torch.FloatTensor(y_scaled).to(device)
        pred_pca = gen_model(t).cpu().numpy()
    pred_x_scaled = gen_pca.inverse_transform(pred_pca)
    return gen_x_scaler.inverse_transform(pred_x_scaled)


def find_best_sample(x_test_pca, y_test, char_model, char_y_scaler, device):
    """Return the index of the test sample with lowest characterization RMSE."""
    with torch.no_grad():
        pred_scaled = (
            char_model(torch.FloatTensor(x_test_pca).to(device)).cpu().numpy()
        )
    pred_params = char_y_scaler.inverse_transform(pred_scaled)
    per_sample_rmse = np.sqrt(np.mean((y_test - pred_params) ** 2, axis=1))
    best_idx = int(np.argmin(per_sample_rmse))
    print(
        f"  Best sample: index {best_idx}, char RMSE = {per_sample_rmse[best_idx]:.6f}"
    )
    return best_idx


# ---------------------------------------------------------------------------
# Plot 1: LC Reconstruction
# ---------------------------------------------------------------------------


def plot_lc_reconstruction(
    sample_idx,
    x_test_clean,
    char_model,
    char_x_scaler,
    char_pca,
    gen_model,
    gen_x_scaler,
    gen_pca,
    n_days,
    samples_per_day,
    noise_std,
    device,
    lsst_seed=42,
):
    """
    3-element overlay plot for a single light curve:
      Original  (blue solid)
      Desampled (green dots)
      Noisy     (black markers + errorbars)
      Reconstructed (red dashed)
    """
    time_axis = np.arange(n_days) / samples_per_day  # days from explosion
    original = x_test_clean[sample_idx]

    # --- LSST cadence mask (deterministic) ---
    obs_idx = get_lsst_observed_idx(n_days, samples_per_day, seed=lsst_seed)

    # Desampled: original values at observed epochs (no noise)
    desampled_times = time_axis[obs_idx]
    desampled_vals = original[obs_idx]

    # Noisy: add Gaussian noise at observed epochs
    rng = np.random.default_rng(lsst_seed)
    noise = rng.normal(0.0, noise_std, size=len(obs_idx))
    noisy_vals = desampled_vals + noise

    # Reconstructed: augmented curve -> char -> gen
    aug_curve = np.interp(np.arange(n_days), obs_idx, noisy_vals)
    aug_pca = char_pca.transform(
        char_x_scaler.transform(aug_curve[np.newaxis, :])
    )
    with torch.no_grad():
        pred_params_scaled = (
            char_model(torch.FloatTensor(aug_pca).to(device)).cpu().numpy()
        )
    reconstructed = generate_curves(
        pred_params_scaled, gen_model, gen_x_scaler, gen_pca, device
    )[0]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(time_axis, original, "b-", linewidth=1.5, label="Original")
    ax.plot(
        desampled_times, desampled_vals, "g.", markersize=5, label="Desampled"
    )
    ax.plot(desampled_times, noisy_vals, "k.", markersize=3, label="Noisy")
    ax.errorbar(
        desampled_times,
        noisy_vals,
        yerr=noise_std,
        fmt="none",
        ecolor="black",
        elinewidth=0.5,
        capsize=0,
        label="Error",
    )
    # ax.axvline(x=0, color='r', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.plot(
        time_axis, reconstructed, "r--", linewidth=1.5, label="Reconstructed"
    )
    # ax.axvline(x=0, color='r', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Epochs [days from explosion epoch]")
    ax.set_ylabel("Luminosity\nlog$_{10}$(L$_{bol}$[erg/s])")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Per-timestep Reconstruction Error
# ---------------------------------------------------------------------------


def compute_chargen_rmse(
    x_test_clean,
    x_test_aug,
    char_model,
    char_x_scaler,
    char_pca,
    gen_model,
    gen_x_scaler,
    gen_pca,
    device,
):
    """Full pipeline RMSE per timestep: aug curve -> char -> gen -> reconstructed.
    Returns RMSE at each time step across all test samples.
    x_test_clean: (n_test, n_timepoints) original clean curves
    x_test_aug: (n_test, n_timepoints) augmented curves (LSST cadence + noise)
    char_model: trained characterizer model
    char_x_scaler: characterizer input scaler
    char_pca: characterizer input PCA
    gen_model: trained generator model
    gen_x_scaler: generator input scaler
    gen_pca: generator input PCA
    device: torch.device to run on"""
    x_aug_pca = char_pca.transform(char_x_scaler.transform(x_test_aug))
    with torch.no_grad():
        pred_params_scaled = (
            char_model(torch.FloatTensor(x_aug_pca).to(device)).cpu().numpy()
        )
    pred_curves = generate_curves(
        pred_params_scaled, gen_model, gen_x_scaler, gen_pca, device
    )
    return np.sqrt(np.mean((x_test_clean - pred_curves) ** 2, axis=0))


def plot_reconstruction_error(
    x_test_clean,
    x_test_aug,
    y_test_scaled,
    char_model,
    char_x_scaler,
    char_pca,
    gen_model,
    gen_x_scaler,
    gen_pca,
    n_days,
    samples_per_day,
    device,
):
    """
    Log-scale RMSE at each time step for three scenarios:
      1. Generator only      — teacher-forced true params → gen → curve
      2. Char → Gen          — aug curve → char → params → gen → curve
      3. Augmentation        — LSST-augmented curve vs original (no model)
    """
    time_axis = np.arange(n_days) / samples_per_day

    # 1. Generator only (teacher forcing — lower bound)
    pred_gen = generate_curves(
        y_test_scaled, gen_model, gen_x_scaler, gen_pca, device
    )
    rmse_gen = np.sqrt(np.mean((x_test_clean - pred_gen) ** 2, axis=0))

    # 2. Char → Gen on clean input (no LSST degradation)
    rmse_chargen_clean = compute_chargen_rmse(
        x_test_clean,
        x_test_clean,
        char_model,
        char_x_scaler,
        char_pca,
        gen_model,
        gen_x_scaler,
        gen_pca,
        device,
    )

    # 3. Char → Gen on augmented input (realistic LSST scenario)
    rmse_chargen_aug = compute_chargen_rmse(
        x_test_clean,
        x_test_aug,
        char_model,
        char_x_scaler,
        char_pca,
        gen_model,
        gen_x_scaler,
        gen_pca,
        device,
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(
        time_axis,
        rmse_chargen_aug,
        linewidth=1.2,
        label="Reconstruction error + Noise",
    )
    ax.semilogy(
        time_axis,
        rmse_chargen_clean,
        linewidth=1.2,
        label="Reconstruction error",
    )
    ax.semilogy(time_axis, rmse_gen, linewidth=1.2, label="Generation error")

    ax.set_xlabel("Epochs [days from explosion epoch]")
    ax.set_ylabel("Luminosity\nlog$_{10}$(L$_{bol}$[erg/s]) RMSE")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Main function to generate ASTRAI result plots.
    Loads characterizer and generator models from specified experiment directories,
    loads test data from the specified fold, and creates two plots:
        1. LC Reconstruction for the best sample by characterization RMSE.
        2. Per-timestep reconstruction error across the test set.
        Saves plots to the specified output directory."""
    parser = argparse.ArgumentParser(
        description="Generate ASTRAI result plots."
    )
    parser.add_argument(
        "--config",
        default="configs/default_split.yaml",
        help="Path to split config YAML",
    )
    parser.add_argument(
        "--exp_char", required=True, help="Characterizer experiment directory"
    )
    parser.add_argument(
        "--exp_gen", required=True, help="Generator experiment directory"
    )
    parser.add_argument(
        "--prep", default="preprocessed", help="Preprocessed data directory"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold to use for test data (default: 1)",
    )
    parser.add_argument(
        "--output_dir", default=".", help="Directory where plots are saved"
    )
    parser.add_argument(
        "--lsst_seed",
        type=int,
        default=42,
        help="RNG seed for the LSST mask in Plot 1 (default: 42)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_days = cfg["data"]["n_days"]
    samples_per_day = cfg["data"].get("samples_per_day", 4)
    noise_std = cfg["augmentation"]["noise_std"]

    # Load models
    print("Loading characterizer...")
    char_model, char_x_scaler, char_y_scaler, char_pca = load_characterizer(
        cfg, device, args.exp_char
    )

    print("Loading generator...")
    gen_model, gen_x_scaler, _, gen_pca = load_generator(
        cfg, device, args.exp_gen
    )

    # Load test data from chosen fold
    fold_dir = os.path.join(args.prep, f"fold_{args.fold}")
    print(f"Loading test data from: {fold_dir}")
    x_test_clean = np.load(os.path.join(fold_dir, "x_test_clean.npy"))
    x_test_pca = np.load(os.path.join(fold_dir, "x_test_pca.npy"))
    y_test = np.load(os.path.join(fold_dir, "y_test.npy"))
    y_test_scaled = np.load(os.path.join(fold_dir, "y_test_scaled.npy"))

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Plot 1: LC Reconstruction ---
    print("\n[Plot 1] Finding best sample by characterization RMSE...")
    best_idx = find_best_sample(
        x_test_pca, y_test, char_model, char_y_scaler, device
    )

    fig1 = plot_lc_reconstruction(
        best_idx,
        x_test_clean,
        char_model,
        char_x_scaler,
        char_pca,
        gen_model,
        gen_x_scaler,
        gen_pca,
        n_days,
        samples_per_day,
        noise_std,
        device,
        lsst_seed=args.lsst_seed,
    )
    out1 = os.path.join(args.output_dir, "lc_reconstruction.pdf")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {out1}")

    # --- Plot 2: Reconstruction Error ---
    print("\n[Plot 2] Applying LSST augmentation to test set...")
    x_test_aug = apply_lsst_pipeline(
        x_test_clean, n_days, noise_std, samples_per_day=samples_per_day
    )

    print("[Plot 2] Computing per-timestep reconstruction error (3 curves)...")
    fig2 = plot_reconstruction_error(
        x_test_clean,
        x_test_aug,
        y_test_scaled,
        char_model,
        char_x_scaler,
        char_pca,
        gen_model,
        gen_x_scaler,
        gen_pca,
        n_days,
        samples_per_day,
        device,
    )
    out2 = os.path.join(args.output_dir, "reconstruction_error.pdf")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {out2}")

    plt.show()


if __name__ == "__main__":
    main()
