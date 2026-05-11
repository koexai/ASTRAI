"""
infer_bol_batch.py - Batch inference on all supernovae in the bol/ directory.

Reads explosion epochs from bol/info_87Alike.txt, matches each entry to
the corresponding bol/bol_<SN>_<filters>.txt file, runs characterization
(curves -> params) and generation (params -> curves), and saves one PDF
plot per supernova.

Usage::

    python infer_bol_batch.py
    python infer_bol_batch.py --bol_dir bol --output_dir plots/batch
    python infer_bol_batch.py --exp_char experiments/characterizer/YYYYMMDD_HHMMSS \
                               --exp_gen  experiments/generator/YYYYMMDD_HHMMSS
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from utils.checkpoints import load_config, load_characterizer, load_generator

BOL_DIR = "data/real/87Alike_bolometric"
INFO_FILE = "data/real/metadata/info_87Alike.txt"
CONFIG_PATH = "configs/default_split.yaml"
EXP_CHAR_DEFAULT = "experiments/characterizer/20260330_170653"
EXP_GEN_DEFAULT = "experiments/generator/20260310_171951"
OUTPUT_DIR = "plots/batch"


def load_info(info_path):
    """Return a dict {sn_name_upper: t_explosion} from info_87Alike.txt.
    Expects a whitespace-separated txt file with columns SN_n and t_explosion.
    """
    df = pd.read_csv(info_path, sep=r"\s+", comment=None)
    df.columns = df.columns.str.strip()
    return {
        row["SN_n"].upper(): float(row["t_explosion"])
        for _, row in df.iterrows()
    }


def find_bol_file(sn_name, bol_dir):
    """Find the bol txt file matching sn_name (case-insensitive prefix match).
    Expects files named like bol_<SN>_<filters>.txt, e.g. bol_2018hna_UBVRI.txt.
    Returns the first match or None if not found."""
    pattern = os.path.join(bol_dir, f"bol_{sn_name}_*.txt")
    matches = glob.glob(pattern, recursive=False)
    if not matches:
        # Try case-insensitive fallback
        all_files = glob.glob(os.path.join(bol_dir, "bol_*.txt"))
        matches = [
            f
            for f in all_files
            if os.path.basename(f).upper().startswith(f"BOL_{sn_name.upper()}_")
        ]
    return matches[0] if matches else None


def load_bol_txt(txt_path, explosion_mjd, n_days):
    """Read a bol txt file and interpolate to a uniform [0, n_days-1] day grid.

    Format: tab-separated, first line is a comment (# ph Lobs err L+BB err).
    Uses the L+BB column (extinction-corrected bolometric luminosity),
    converts to log10, and linearly interpolates to the model's uniform grid.

    Returns
    -------
    curve : np.ndarray, shape (n_days,)
    obs_days, obs_log10, obs_err_log10 : original sparse observations
    """
    df = pd.read_csv(
        txt_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["ph", "Lobs", "err_Lobs", "L_BB", "err_L_BB"],
    )

    df["rel_day"] = df["ph"] - explosion_mjd

    # Keep only epochs within the model's time window [0, n_days-1]
    df = df[(df["rel_day"] >= 0) & (df["rel_day"] <= n_days - 1)].copy()

    if len(df) < 2:
        raise ValueError(
            f"Too few observations within the {n_days}-day window "
            f"(found {len(df)})"
        )

    df = df.sort_values("rel_day")

    # Average near-duplicate epochs (same night, multiple passes)
    df["day_key"] = df["rel_day"].round(2)
    df_agg = (
        df.groupby("day_key")
        .agg(
            rel_day=("rel_day", "mean"),
            L_BB=("L_BB", "mean"),
            err_L_BB=("err_L_BB", "mean"),
        )
        .reset_index(drop=True)
    )

    obs_days = df_agg["rel_day"].values
    obs_lum = df_agg["L_BB"].values
    obs_err = df_agg["err_L_BB"].values

    obs_log10 = np.log10(obs_lum)
    obs_err_log10 = obs_err / (np.log(10) * obs_lum)  # error propagation

    # Uniform grid: NaN outside the observed range (avoids flat-plateau artifact)
    grid = np.arange(n_days, dtype=float)
    curve = np.interp(grid, obs_days, obs_log10, left=np.nan, right=np.nan)

    return curve, obs_days, obs_log10, obs_err_log10


def run_characterization(curve, char_model, x_scaler, y_scaler, pca, device):
    """Curve (1D array) -> predicted physical parameters (original scale).
    curve: (n_timepoints,) input light curve (log10 L_bol)
    char_model: the loaded characterizer model
    x_scaler, y_scaler, pca: preprocessing artifacts for scaling and PCA
    device: torch.device to run on
    Returns:
    pred_params: (n_params,) predicted physical parameters on original scale
    pred_sc: (n_params,) predicted parameters on the characterizer's scaled PCA space
    (useful for feeding into the generator)
    """
    x = curve[np.newaxis, :].astype("float32")
    x_scaled = x_scaler.transform(x)
    # Fill NaN (outside observed range) via default np.interp edge clamping
    x1d = x_scaled[0]
    valid = np.where(~np.isnan(x1d))[0]
    x_scaled[0] = np.interp(np.arange(len(x1d)), valid, x1d[valid])
    x_pca = pca.transform(x_scaled)

    with torch.no_grad():
        pred_sc = char_model(torch.FloatTensor(x_pca).to(device)).cpu().numpy()

    pred_log1p = y_scaler.inverse_transform(pred_sc)[0]
    pred_params = np.expm1(pred_log1p)
    return pred_params, pred_sc


def run_generation(pred_sc, gen_model, x_scaler, pca, device):
    """Predicted params (scaled) -> reconstructed curve (log10 L_bol).
    pred_sc: (1, n_params) scaled parameters from the characterizer
    Returns: (n_timepoints,) reconstructed curve with inverse PCA and scaling applied.
    """
    with torch.no_grad():
        pred_pca = (
            gen_model(torch.FloatTensor(pred_sc).to(device)).cpu().numpy()
        )

    return x_scaler.inverse_transform(pca.inverse_transform(pred_pca))[0]


# ---------------------------------------------------------------------------
# Plot  (identical to infer_sn2018hna.py, title shows SN name)
# ---------------------------------------------------------------------------


def make_plot(
    _sn_name,
    explosion_mjd,
    time_axis,
    curve_input,
    reconstructed,
    obs_days,
    obs_log10,
    obs_err_log10,
    param_names,
    pred_params,
    output_path=None,
):
    """Make the 2-panel plot for a single supernova."""
    fig, axes = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax = axes[0]
    ax.plot(
        time_axis,
        curve_input,
        color="steelblue",
        lw=1.2,
        label="Interpolated input (log10 L+BB)",
    )
    ax.plot(
        time_axis,
        reconstructed,
        color="crimson",
        lw=1.5,
        ls="--",
        label="Model reconstruction (Char→Gen)",
    )
    ax.errorbar(
        obs_days,
        obs_log10,
        yerr=obs_err_log10,
        fmt="o",
        color="black",
        ms=3,
        lw=0.7,
        capsize=2,
        zorder=5,
        label="Observations (L+BB)",
    )

    ax.set_ylabel(r"log$_{10}$(L$_{\rm bol}$ [erg/s])")
    # ax.set_title(f"{sn_name} — Model Inference  (t$_{{exp}}$ = MJD {explosion_mjd})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    param_lines = "\n".join(
        f"{n}: {v:.3g}" for n, v in zip(param_names, pred_params)
    )
    ax.text(
        0.98,
        0.97,
        param_lines,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "alpha": 0.8},
    )

    ax2 = axes[1]
    ax2.plot(time_axis, curve_input - reconstructed, color="gray", lw=0.9)
    ax2.axhline(0, color="k", lw=0.7, ls="--")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel(f"Days from explosion (MJD {explosion_mjd})")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=600, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Main function to run batch inference on real bolometric data."""
    parser = argparse.ArgumentParser(
        description="Batch inference on real bolometric data."
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--bol_dir", default=BOL_DIR)
    parser.add_argument("--info", default=INFO_FILE)
    parser.add_argument("--exp_char", default=EXP_CHAR_DEFAULT)
    parser.add_argument("--exp_gen", default=EXP_GEN_DEFAULT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_days = cfg["data"]["n_days"]
    param_names = cfg["data"]["param_names"]
    samples_per_day = cfg["data"].get("samples_per_day", 1)
    time_axis = np.arange(n_days, dtype=float) / samples_per_day

    # Load explosion epochs
    info = load_info(args.info)
    print(f"Loaded {len(info)} entries from {args.info}")

    # Load models once
    print(f"\nLoading characterizer from: {args.exp_char}")
    char_model, c_xsc, c_ysc, c_pca = load_characterizer(
        cfg, device, args.exp_char
    )
    print(f"Loading generator from: {args.exp_gen}")
    gen_model, g_xsc, _, g_pca = load_generator(cfg, device, args.exp_gen)

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    skipped = []

    for sn_name, t_exp in sorted(info.items()):
        txt_path = find_bol_file(sn_name, args.bol_dir)
        if txt_path is None:
            skipped.append((sn_name, "no bol file found"))
            continue

        print(f"\n{'='*55}")
        print(
            f"  {sn_name}  |  t_exp = {t_exp}  |  {os.path.basename(txt_path)}"
        )
        print(f"{'='*55}")

        try:
            curve, obs_days, obs_log10, obs_err_log10 = load_bol_txt(
                txt_path, t_exp, n_days
            )
        except ValueError as e:
            skipped.append((sn_name, str(e)))
            print(f"  SKIPPED: {e}")
            continue

        print(
            f"  Observations: {len(obs_days)} pts  "
            f"(days {obs_days[0]:.1f}–{obs_days[-1]:.1f})  "
            f"log10 L range [{curve.min():.3f}, {curve.max():.3f}]"
        )

        pred_params, pred_sc = run_characterization(
            curve, char_model, c_xsc, c_ysc, c_pca, device
        )
        reconstructed = run_generation(pred_sc, gen_model, g_xsc, g_pca, device)

        print("  Predicted parameters:")
        for name, val in zip(param_names, pred_params):
            print(f"    {name:<10}: {val:.4g}")

        out_path = os.path.join(args.output_dir, f"{sn_name}_inference.pdf")
        make_plot(
            sn_name,
            t_exp,
            time_axis,
            curve,
            reconstructed,
            obs_days,
            obs_log10,
            obs_err_log10,
            param_names,
            pred_params,
            output_path=out_path,
        )
        plt.close("all")
        print(f"  Plot saved -> {out_path}")

        results.append({"SN": sn_name, **dict(zip(param_names, pred_params))})

    # Summary
    print(f"\n{'='*55}")
    print(f"DONE: {len(results)} processed, {len(skipped)} skipped")
    if skipped:
        print("Skipped:")
        for sn, reason in skipped:
            print(f"  {sn}: {reason}")

    if results:
        df_res = pd.DataFrame(results)
        csv_out = os.path.join(args.output_dir, "batch_parameters.csv")
        df_res.to_csv(csv_out, index=False)
        print(f"\nAll parameters saved to: {csv_out}")
        print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
