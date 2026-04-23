"""
infer_sn2018hna.py - Inference on SN2018hna real observations.

Reads sparse bolometric luminosity measurements, resamples to the uniform
421-day grid expected by the model, runs characterization (curves -> params)
and generation (params -> curves), then plots the result.

Usage::

    python infer_sn2018hna.py
    python infer_sn2018hna.py --exp_char experiments/characterizer/YYYYMMDD_HHMMSS \
                               --exp_gen  experiments/generator/YYYYMMDD_HHMMSS
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import joblib
import yaml
import matplotlib.pyplot as plt

from astrai.models import SplitMLPRegressor, MLPWithResiduals

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPLOSION_EPOCH_MJD = 58413.82
CSV_PATH = "SN2018hna.csv"
CONFIG_PATH = "configs/default_split.yaml"
EXP_CHAR_DEFAULT = "experiments/characterizer/20260310_171704"
EXP_GEN_DEFAULT  = "experiments/generator/20260310_171951"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_sn_csv(csv_path, explosion_mjd, n_days):
    """Read the SN CSV and interpolate to a uniform [0, n_days-1] day grid.

    Uses lum_corr (extinction-corrected bolometric luminosity) and converts
    to log10 to match the training data format.  Sparse observations are
    linearly interpolated; epochs before the first detection are filled with
    the first detected value (edge clamping).

    Returns
    -------
    curve : np.ndarray, shape (n_days,)
        log10(L_bol) on a uniform daily grid.
    obs_days : np.ndarray
        Relative days of the original (de-duplicated) observations.
    obs_log10 : np.ndarray
        log10(lum_corr) at the original observation epochs.
    obs_err_log10 : np.ndarray
        Approximate log10-space error (delta_log10 ≈ err / (ln10 * lum)).
    """
    df = pd.read_csv(csv_path, sep=";")
    df["rel_day"] = df["Epoch"] - explosion_mjd

    # Keep only epochs within the model's time window
    df = df[(df["rel_day"] >= 0) & (df["rel_day"] <= n_days - 1)].copy()
    df = df.sort_values("rel_day")

    # Average near-duplicate epochs (same day, different filter passes)
    df["day_int"] = df["rel_day"].round(2)
    df_agg = df.groupby("day_int").agg(
        rel_day=("rel_day", "mean"),
        lum_corr=("lum_corr", "mean"),
        err_lum_corr=("err_lum_corr", "mean"),
    ).reset_index(drop=True)

    obs_days = df_agg["rel_day"].values
    obs_lum  = df_agg["lum_corr"].values
    obs_err  = df_agg["err_lum_corr"].values

    obs_log10     = np.log10(obs_lum)
    obs_err_log10 = obs_err / (np.log(10) * obs_lum)  # error propagation

    # Uniform grid: 0, 1, 2, ..., n_days-1
    grid = np.arange(n_days, dtype=float)
    curve = np.interp(grid, obs_days, obs_log10)  # edge-clamps automatically

    return curve, obs_days, obs_log10, obs_err_log10


def load_characterizer(cfg, device, exp_dir):
    char_cfg = cfg["characterizer"]
    n_pca    = cfg["preprocessing"]["pca_components"]
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
        os.path.join(exp_dir, ckpt["model"]),
        map_location=device, weights_only=True,
    ))
    model.eval()

    x_scaler = joblib.load(os.path.join(exp_dir, ckpt["x_scaler"]))
    y_scaler = joblib.load(os.path.join(exp_dir, ckpt["y_scaler"]))
    pca      = joblib.load(os.path.join(exp_dir, ckpt["pca"]))
    return model, x_scaler, y_scaler, pca


def load_generator(cfg, device, exp_dir):
    gen_cfg  = cfg["generator"]
    n_pca    = cfg["preprocessing"]["pca_components"]
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
        os.path.join(exp_dir, ckpt["model"]),
        map_location=device, weights_only=True,
    ))
    model.eval()

    x_scaler = joblib.load(os.path.join(exp_dir, ckpt["x_scaler"]))
    y_scaler = joblib.load(os.path.join(exp_dir, ckpt["y_scaler"]))
    pca      = joblib.load(os.path.join(exp_dir, ckpt["pca"]))
    return model, x_scaler, y_scaler, pca


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_characterization(curve, char_model, x_scaler, y_scaler, pca, device):
    """Curve (1D array) -> predicted physical parameters (original scale)."""
    X = curve[np.newaxis, :].astype("float32")      # (1, n_days)
    X_scaled = x_scaler.transform(X)
    X_pca    = pca.transform(X_scaled)

    with torch.no_grad():
        pred_sc = char_model(torch.FloatTensor(X_pca).to(device)).cpu().numpy()

    # Inverse-transform: y_scaler reverses StandardScaler, expm1 reverses log1p
    pred_log1p = y_scaler.inverse_transform(pred_sc)[0]
    pred_params = np.expm1(pred_log1p)
    return pred_params, pred_sc  # pred_sc needed for generator input


def run_generation(pred_sc, gen_model, x_scaler, y_scaler, pca, device):
    """Predicted params (scaled) -> reconstructed curve (log10 L_bol)."""
    with torch.no_grad():
        pred_pca = gen_model(torch.FloatTensor(pred_sc).to(device)).cpu().numpy()

    # Inverse PCA + inverse StandardScaler -> log10 L_bol
    reconstructed = x_scaler.inverse_transform(pca.inverse_transform(pred_pca))[0]
    return reconstructed


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(
    time_axis, curve_input, reconstructed,
    obs_days, obs_log10, obs_err_log10,
    param_names, pred_params,
    n_days, samples_per_day,
    output_path=None,
):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)

    # --- Top panel: light curves ---
    ax = axes[0]
    ax.plot(time_axis, curve_input, color="steelblue", lw=1.2,
            label="Interpolated input (log10 lum_corr)")
    ax.plot(time_axis, reconstructed, color="crimson", lw=1.5, ls="--",
            label="Model reconstruction (Char→Gen)")
    ax.errorbar(obs_days, obs_log10, yerr=obs_err_log10,
                fmt="o", color="black", ms=3, lw=0.7, capsize=2, zorder=5,
                label="Observations (lum_corr)")

    ax.set_ylabel(r"log$_{10}$(L$_{\rm bol}$ [erg/s])")
    ax.set_title("SN2018hna — Model Inference")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Parameter box
    param_lines = "\n".join(
        f"{n}: {v:.3g}" for n, v in zip(param_names, pred_params)
    )
    ax.text(0.98, 0.97, param_lines,
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    # --- Bottom panel: residuals (input - reconstructed) ---
    ax2 = axes[1]
    residual = curve_input - reconstructed
    ax2.plot(time_axis, residual, color="gray", lw=0.9)
    ax2.axhline(0, color="k", lw=0.7, ls="--")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel(f"Days from explosion (MJD {EXPLOSION_EPOCH_MJD})")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference on SN2018hna.")
    parser.add_argument("--config",   default=CONFIG_PATH)
    parser.add_argument("--csv",      default=CSV_PATH)
    parser.add_argument("--exp_char", default=EXP_CHAR_DEFAULT)
    parser.add_argument("--exp_gen",  default=EXP_GEN_DEFAULT)
    parser.add_argument("--output",   default="sn2018hna_inference.pdf",
                        help="Output plot path (pdf/png)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_days        = cfg["data"]["n_days"]
    samples_per_day = cfg["data"].get("samples_per_day", 1)
    param_names   = cfg["data"]["param_names"]

    # --- 1. Load and preprocess the SN data ---
    print(f"\nLoading SN2018hna from: {args.csv}")
    curve, obs_days, obs_log10, obs_err_log10 = load_sn_csv(
        args.csv, EXPLOSION_EPOCH_MJD, n_days)

    time_axis = np.arange(n_days) / samples_per_day
    print(f"  Observations used: {len(obs_days)} points "
          f"(days {obs_days[0]:.1f} – {obs_days[-1]:.1f})")
    print(f"  Interpolated grid: {n_days} points "
          f"(log10 L range {curve.min():.3f} – {curve.max():.3f})")

    # --- 2. Load models ---
    print(f"\nLoading characterizer from: {args.exp_char}")
    char_model, c_xsc, c_ysc, c_pca = load_characterizer(cfg, device, args.exp_char)

    print(f"Loading generator from: {args.exp_gen}")
    gen_model, g_xsc, g_ysc, g_pca = load_generator(cfg, device, args.exp_gen)

    # --- 3. Characterization ---
    print("\nRunning characterization...")
    pred_params, pred_sc = run_characterization(curve, char_model, c_xsc, c_ysc, c_pca, device)

    print("\n--- Predicted physical parameters ---")
    for name, val in zip(param_names, pred_params):
        print(f"  {name:<10}: {val:.4g}")

    # --- 4. Generation ---
    print("\nRunning generation...")
    reconstructed = run_generation(pred_sc, gen_model, g_xsc, g_ysc, g_pca, device)

    # --- 5. Plot ---
    print("\nGenerating plot...")
    fig = make_plot(
        time_axis, curve, reconstructed,
        obs_days, obs_log10, obs_err_log10,
        param_names, pred_params,
        n_days, samples_per_day,
        output_path=args.output,
    )
    plt.show()


if __name__ == "__main__":
    main()
