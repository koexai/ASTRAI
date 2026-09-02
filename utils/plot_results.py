"""Generate reproducible diagnostics for compatible ASTRAI experiments.

The command validates the characterizer and generator configurations,
their held-out fold, their shared physical-parameter scaling, and the
preprocessed test artefacts. It applies one seeded LSST augmentation to
the complete test batch and produces:

1. light-curve reconstructions for selected test samples;
2. clean-versus-augmented input diagnostics for those samples;
3. per-timestep reconstruction RMSE curves;
4. CSV files containing the numerical values shown in the figures.

Named samples are ranked exclusively by augmentation RMSE, not by model
performance. Run this module from the repository root with
``python -m utils.plot_results``.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.augmentation import apply_lsst_pipeline
from utils.checkpoints import load_config, load_characterizer, load_generator


_SAMPLE_DIAGNOSTIC_HEADER = (
    "time_days,clean_target,augmented_model_input,"
    "retained_mask,model_reconstruction"
)
_SAMPLE_DIAGNOSTIC_FORMAT = (
    "%.18e",
    "%.18e",
    "%.18e",
    "%d",
    "%.18e",
)
_RECONSTRUCTION_ERROR_HEADER = (
    "time_days,generator_true_parameters_rmse,"
    "full_pipeline_clean_input_rmse,"
    "full_pipeline_augmented_input_rmse"
)
_RECONSTRUCTION_ERROR_FORMAT = ("%.18e",) * 4

_SHARED_CONFIG_FIELDS = (
    ("data.format", ("data", "format")),
    ("data.path", ("data", "path")),
    ("data.curves_path", ("data", "curves_path")),
    ("data.params_path", ("data", "params_path")),
    ("data.params_csv_sep", ("data", "params_csv_sep")),
    ("data.n_days", ("data", "n_days")),
    ("data.n_params", ("data", "n_params")),
    ("data.param_names", ("data", "param_names")),
    ("data.samples_per_day", ("data", "samples_per_day")),
    (
        "preprocessing.pca_components",
        ("preprocessing", "pca_components"),
    ),
    ("preprocessing.n_splits", ("preprocessing", "n_splits")),
    (
        "preprocessing.random_seed",
        ("preprocessing", "random_seed"),
    ),
    ("augmentation.noise_std", ("augmentation", "noise_std")),
)


def _config_value(cfg, path):
    """Return a nested configuration value, or None if absent."""
    value = cfg

    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]

    return value


def load_experiment_config(exp_dir):
    """Load the single YAML configuration in an experiment directory."""
    exp_path = Path(exp_dir)
    config_paths = sorted(exp_path.glob("*.yaml"))
    config_paths.extend(sorted(exp_path.glob("*.yml")))

    if len(config_paths) != 1:
        raise ValueError(
            f"Expected exactly one YAML configuration in {exp_path}, "
            f"found {len(config_paths)}."
        )

    config_path = config_paths[0]
    return load_config(config_path), config_path


def validate_experiment_configs(char_cfg, gen_cfg):
    """Ensure that the two experiments use compatible configurations."""
    mismatches = []

    for label, path in _SHARED_CONFIG_FIELDS:
        char_value = _config_value(char_cfg, path)
        gen_value = _config_value(gen_cfg, path)

        if char_value != gen_value:
            mismatches.append(
                f"{label}: characterizer={char_value!r}, "
                f"generator={gen_value!r}"
            )

    if mismatches:
        details = "\n  - ".join(mismatches)
        raise ValueError(
            "Incompatible characterizer and generator configurations:\n"
            f"  - {details}"
        )


def validate_shared_parameter_scaling(
    char_y_scaler,
    gen_y_scaler,
):
    """Ensure PPReg outputs and LCGen inputs use the same scaling.

    PPReg predicts physical parameters in standardised ``y`` space,
    while LCGen consumes parameters in that same space. Different
    scalers are therefore considered an incompatible pipeline rather
    than being converted automatically.
    """
    if type(char_y_scaler) is not type(gen_y_scaler):
        raise ValueError(
            "Incompatible parameter scaling: the characterizer and "
            "generator use different scaler types."
        )

    for attribute in ("mean_", "scale_"):
        if not hasattr(char_y_scaler, attribute):
            raise ValueError(
                "The characterizer parameter scaler does not expose "
                f"{attribute}."
            )

        if not hasattr(gen_y_scaler, attribute):
            raise ValueError(
                "The generator parameter scaler does not expose "
                f"{attribute}."
            )

        char_value = np.asarray(
            getattr(char_y_scaler, attribute),
            dtype=float,
        )
        gen_value = np.asarray(
            getattr(gen_y_scaler, attribute),
            dtype=float,
        )

        if (
            char_value.shape != gen_value.shape
            or not np.allclose(
                char_value,
                gen_value,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
        ):
            raise ValueError(
                "Incompatible parameter scaling: characterizer output "
                f"and generator input {attribute} values differ."
            )


def validate_scaled_parameter_artifact(
    y_test,
    y_test_scaled,
    parameter_scaler,
):
    """Ensure the stored scaled parameters match the shared scaler."""
    if y_test.shape != y_test_scaled.shape:
        raise ValueError(
            "y_test and y_test_scaled must have the same shape."
        )

    expected_y_test_scaled = parameter_scaler.transform(y_test)

    if not np.allclose(
        y_test_scaled,
        expected_y_test_scaled,
        rtol=1e-6,
        atol=1e-7,
        equal_nan=False,
    ):
        raise ValueError(
            "The stored y_test_scaled artefact is incompatible with "
            "the shared parameter scaler."
        )


def resolve_diagnostic_fold(
    char_cfg,
    gen_cfg,
    requested_fold,
    characterizer_fold=None,
):
    """Validate the fold used by all diagnostic inputs.

    characterizer_fold supplies the missing metadata for legacy PPReg
    experiments that did not record held_out_fold in their configuration.
    """
    configured_char_fold = _config_value(
        char_cfg,
        ("characterizer", "training", "held_out_fold"),
    )
    configured_gen_fold = _config_value(
        gen_cfg,
        ("generator", "training", "held_out_fold"),
    )

    if configured_char_fold is None:
        if characterizer_fold is None:
            raise ValueError(
                "The characterizer experiment does not record "
                "held_out_fold; pass --characterizer-fold explicitly."
            )
        resolved_char_fold = characterizer_fold
    else:
        resolved_char_fold = configured_char_fold

        if (
            characterizer_fold is not None
            and characterizer_fold != configured_char_fold
        ):
            raise ValueError(
                "The --characterizer-fold value does not match the "
                "characterizer experiment configuration."
            )

    if configured_gen_fold is None:
        raise ValueError(
            "The generator experiment does not record held_out_fold."
        )

    n_splits = _config_value(
        char_cfg,
        ("preprocessing", "n_splits"),
    )
    if n_splits is not None and not 1 <= requested_fold <= n_splits:
        raise ValueError(
            f"Requested fold {requested_fold} is outside the valid "
            f"range 1-{n_splits}."
        )

    folds = {
        "characterizer": resolved_char_fold,
        "generator": configured_gen_fold,
        "preprocessed data": requested_fold,
    }

    if len(set(folds.values())) != 1:
        details = ", ".join(
            f"{source}={fold}" for source, fold in folds.items()
        )
        raise ValueError(f"Fold mismatch: {details}.")

    return requested_fold


def select_diagnostic_samples(
    x_test_clean,
    x_test_aug,
    selections=None,
):
    """Select samples according to the effect of augmentation.

    The ranking is based exclusively on the RMSE between each clean curve
    and its augmented counterpart. It does not measure model performance.

    Supported selections are ``representative``, ``least-affected``,
    ``most-affected``, and explicit zero-based test-set indices.
    """
    if x_test_clean.shape != x_test_aug.shape:
        raise ValueError(
            "Clean and augmented curves must have the same shape."
        )

    if x_test_clean.ndim != 2 or x_test_clean.shape[0] == 0:
        raise ValueError(
            "Expected non-empty two-dimensional batches of curves."
        )

    augmentation_rmse = np.sqrt(
        np.mean((x_test_clean - x_test_aug) ** 2, axis=1)
    )

    if not selections:
        selections = ["representative"]

    median_rmse = np.median(augmentation_rmse)
    named_indices = {
        "representative": int(
            np.argmin(np.abs(augmentation_rmse - median_rmse))
        ),
        "least-affected": int(np.argmin(augmentation_rmse)),
        "most-affected": int(np.argmax(augmentation_rmse)),
    }

    selected = []
    seen = set()

    for selection in selections:
        normalised = str(selection).strip().lower()

        if normalised in named_indices:
            sample_idx = named_indices[normalised]
            output_name = normalised.replace("-", "_")
        else:
            try:
                sample_idx = int(normalised)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown sample selection {selection!r}. Use "
                    "representative, least-affected, most-affected, "
                    "or a zero-based test-set index."
                ) from exc

            if not 0 <= sample_idx < len(augmentation_rmse):
                raise ValueError(
                    f"Sample index {sample_idx} is out of range for "
                    f"{len(augmentation_rmse)} test samples."
                )

            output_name = f"sample_{sample_idx}"

        selection_key = (output_name, sample_idx)
        if selection_key in seen:
            continue

        seen.add(selection_key)
        selected.append(
            {
                "name": output_name,
                "index": sample_idx,
                "augmentation_rmse": float(
                    augmentation_rmse[sample_idx]
                ),
            }
        )

    return selected


def save_sample_diagnostic_csv(
    output_path,
    time_axis,
    clean_curve,
    augmented_curve,
    retained_mask,
    reconstructed_curve,
):
    """Save the numerical data shown in one sample diagnostic."""
    diagnostic_data = np.column_stack(
        (
            time_axis,
            clean_curve,
            augmented_curve,
            np.asarray(retained_mask, dtype=np.uint8),
            reconstructed_curve,
        )
    )

    np.savetxt(
        output_path,
        diagnostic_data,
        delimiter=",",
        fmt=_SAMPLE_DIAGNOSTIC_FORMAT,
        header=_SAMPLE_DIAGNOSTIC_HEADER,
        comments="",
    )


def save_reconstruction_error_csv(
    output_path,
    time_axis,
    error_curves,
):
    """Save the numerical data shown in the reconstruction-error plot."""
    error_data = np.column_stack(
        (
            time_axis,
            error_curves["generator_true_parameters_rmse"],
            error_curves["full_pipeline_clean_input_rmse"],
            error_curves["full_pipeline_augmented_input_rmse"],
        )
    )

    np.savetxt(
        output_path,
        error_data,
        delimiter=",",
        fmt=_RECONSTRUCTION_ERROR_FORMAT,
        header=_RECONSTRUCTION_ERROR_HEADER,
        comments="",
    )


def generate_curves(y_scaled, gen_model, gen_x_scaler, gen_pca, device):
    """Generate curves in the original light-curve space.

    ``y_scaled`` contains standardised physical parameters. The generator
    predicts PCA coefficients in scaled light-curve space; ``gen_pca`` and
    ``gen_x_scaler`` then decode those predictions into light curves.
    """
    with torch.no_grad():
        t = torch.FloatTensor(y_scaled).to(device)
        pred_pca = gen_model(t).cpu().numpy()
    pred_x_scaled = gen_pca.inverse_transform(pred_pca)
    return gen_x_scaler.inverse_transform(pred_x_scaled)


# ---------------------------------------------------------------------------
# Plot 1: LC Reconstruction
# ---------------------------------------------------------------------------


def plot_lc_reconstruction(
    sample_idx,
    x_test_clean,
    x_test_aug,
    retained_mask,
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
    """Plot clean target, retained observations, and model reconstruction."""
    time_axis = np.arange(n_days) / samples_per_day
    clean_curve = x_test_clean[sample_idx]
    augmented_curve = x_test_aug[sample_idx]
    sample_mask = retained_mask[sample_idx]

    augmented_pca = char_pca.transform(
        char_x_scaler.transform(augmented_curve[np.newaxis, :])
    )

    with torch.no_grad():
        predicted_params_scaled = (
            char_model(
                torch.FloatTensor(augmented_pca).to(device)
            )
            .cpu()
            .numpy()
        )

    reconstructed_curve = generate_curves(
        predicted_params_scaled,
        gen_model,
        gen_x_scaler,
        gen_pca,
        device,
    )[0]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.plot(
        time_axis,
        clean_curve,
        color="tab:blue",
        linewidth=1.5,
        label="Clean target",
    )
    ax.plot(
        time_axis[sample_mask],
        augmented_curve[sample_mask],
        "k.",
        markersize=4,
        label="Retained noisy samples",
    )
    ax.plot(
        time_axis,
        reconstructed_curve,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Model reconstruction",
    )

    ax.set_xlabel("Epochs [days from explosion epoch]")
    ax.set_ylabel("Luminosity\nlog$_{10}$(L$_{bol}$[erg/s])")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, reconstructed_curve


def plot_augmentation_diagnostic(
    sample_idx,
    x_test_clean,
    x_test_aug,
    retained_mask,
    n_days,
    samples_per_day,
    augmentation_rmse,
):
    """Plot the clean curve, retained samples, and augmented model input."""
    time_axis = np.arange(n_days) / samples_per_day
    clean_curve = x_test_clean[sample_idx]
    augmented_curve = x_test_aug[sample_idx]
    sample_mask = retained_mask[sample_idx]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.plot(
        time_axis,
        clean_curve,
        color="tab:blue",
        linewidth=1.5,
        label="Clean curve",
    )
    ax.plot(
        time_axis,
        augmented_curve,
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label="Augmented model input",
    )
    ax.plot(
        time_axis[sample_mask],
        augmented_curve[sample_mask],
        "k.",
        markersize=4,
        label="Retained noisy samples",
    )

    ax.set_title(
        f"Sample {sample_idx} - augmentation RMSE: "
        f"{augmentation_rmse:.6g}"
    )
    ax.set_xlabel("Epochs [days from explosion epoch]")
    ax.set_ylabel("Luminosity\nlog$_{10}$(L$_{bol}$[erg/s])")
    ax.legend(loc="best", fontsize="small")
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
    """Return full-pipeline RMSE per timestep against clean targets.

    The characterizer consumes ``x_test_aug`` and predicts standardised
    physical parameters. The generator maps those parameters back to PCA
    coefficients in scaled light-curve space. ``gen_pca`` and
    ``gen_x_scaler`` decode the generator output into reconstructed curves.
    """
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
    """Plot per-timestep reconstruction RMSE against clean targets.

    The three scenarios are:

    1. Generator supplied with the true scaled physical parameters.
    2. Full PPReg-LCGen pipeline supplied with clean light curves.
    3. Full PPReg-LCGen pipeline supplied with augmented light curves.
    """
    time_axis = np.arange(n_days) / samples_per_day

    # 1. Generator only with the true scaled parameters
    pred_gen = generate_curves(
        y_test_scaled, gen_model, gen_x_scaler, gen_pca, device
    )
    rmse_gen = np.sqrt(np.mean((x_test_clean - pred_gen) ** 2, axis=0))

    # 2. PPReg-LCGen pipeline on clean input
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

    # 3. PPReg-LCGen pipeline on augmented input
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
        label="Full pipeline (augmented input)",
    )
    ax.semilogy(
        time_axis,
        rmse_chargen_clean,
        linewidth=1.2,
        label="Full pipeline (clean input)",
    )
    ax.semilogy(
        time_axis,
        rmse_gen,
        linewidth=1.2,
        label="Generator only (true parameters)",
    )

    ax.set_xlabel("Epochs [days from explosion epoch]")
    ax.set_ylabel("Reconstruction RMSE [dex]")
    ax.legend(fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    error_curves = {
        "generator_true_parameters_rmse": rmse_gen,
        "full_pipeline_clean_input_rmse": rmse_chargen_clean,
        "full_pipeline_augmented_input_rmse": rmse_chargen_aug,
    }

    return fig, error_curves


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Validate inputs and generate reproducible ASTRAI diagnostics."""
    parser = argparse.ArgumentParser(
        description="Generate ASTRAI result plots."
    )
    parser.add_argument(
        "--exp_char",
        required=True,
        help="Characterizer experiment directory",
    )
    parser.add_argument(
        "--exp_gen",
        required=True,
        help="Generator experiment directory",
    )
    parser.add_argument(
        "--prep",
        required=True,
        help="Preprocessing directory containing the fold_N subdirectories",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold used to select the preprocessed test data",
    )
    parser.add_argument(
        "--characterizer-fold",
        type=int,
        help=(
            "Held-out fold of a legacy characterizer experiment whose "
            "configuration does not record held_out_fold"
        ),
    )
    parser.add_argument(
        "--sample",
        dest="sample_selections",
        action="append",
        help=(
            "Sample to plot: representative, least-affected, "
            "most-affected, or a zero-based test-set index. May be "
            "repeated. Named selections refer only to augmentation effect."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where plots and numerical diagnostics are saved",
    )
    parser.add_argument(
        "--lsst_seed",
        type=int,
        default=42,
        help="RNG seed for the diagnostic augmentation (default: 42)",
    )
    args = parser.parse_args()

    char_cfg, char_config_path = load_experiment_config(args.exp_char)
    gen_cfg, gen_config_path = load_experiment_config(args.exp_gen)

    validate_experiment_configs(char_cfg, gen_cfg)

    diagnostic_fold = resolve_diagnostic_fold(
        char_cfg,
        gen_cfg,
        requested_fold=args.fold,
        characterizer_fold=args.characterizer_fold,
    )

    print(f"Characterizer config: {char_config_path}")
    print(f"Generator config: {gen_config_path}")
    print(f"Diagnostic fold: {diagnostic_fold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_days = char_cfg["data"]["n_days"]
    n_params = char_cfg["data"]["n_params"]
    samples_per_day = char_cfg["data"].get("samples_per_day", 4)
    noise_std = char_cfg["augmentation"]["noise_std"]

    # Load models
    print("Loading characterizer...")
    char_model, char_x_scaler, char_y_scaler, char_pca = load_characterizer(
        char_cfg,
        device,
        args.exp_char,
    )

    print("Loading generator...")
    gen_model, gen_x_scaler, gen_y_scaler, gen_pca = load_generator(
        gen_cfg,
        device,
        args.exp_gen,
    )
    validate_shared_parameter_scaling(
        char_y_scaler,
        gen_y_scaler,
    )
    print("Validated shared PPReg/LCGen parameter scaling.")

    # Load test data from chosen fold
    fold_dir = Path(args.prep) / f"fold_{diagnostic_fold}"
    print(f"Loading test data from: {fold_dir}")

    required_artifacts = {
        "x_test_clean": fold_dir / "x_test_clean.npy",
        "y_test": fold_dir / "y_test.npy",
        "y_test_scaled": fold_dir / "y_test_scaled.npy",
    }
    missing_artifacts = [
        str(path)
        for path in required_artifacts.values()
        if not path.is_file()
    ]

    if missing_artifacts:
        missing_list = "\n  - ".join(missing_artifacts)
        raise FileNotFoundError(
            "Missing preprocessing artefacts. Ensure that --prep points "
            "to the specific preprocessing run directory:\n"
            f"  - {missing_list}"
        )

    x_test_clean = np.load(required_artifacts["x_test_clean"])
    y_test = np.load(required_artifacts["y_test"])
    y_test_scaled = np.load(required_artifacts["y_test_scaled"])

    if x_test_clean.ndim != 2 or x_test_clean.shape[1] != n_days:
        raise ValueError(
            "Unexpected x_test_clean shape: "
            f"{x_test_clean.shape}; expected (n_samples, {n_days})."
        )

    if y_test_scaled.ndim != 2 or y_test_scaled.shape[1] != n_params:
        raise ValueError(
            "Unexpected y_test_scaled shape: "
            f"{y_test_scaled.shape}; expected (n_samples, {n_params})."
        )

    if y_test.ndim != 2 or y_test.shape[1] != n_params:
        raise ValueError(
            "Unexpected y_test shape: "
            f"{y_test.shape}; expected (n_samples, {n_params})."
        )

    if not (
        x_test_clean.shape[0]
        == y_test.shape[0]
        == y_test_scaled.shape[0]
    ):
        raise ValueError(
            "x_test_clean, y_test, and y_test_scaled contain different "
            "numbers of samples."
        )

    validate_scaled_parameter_artifact(
        y_test,
        y_test_scaled,
        char_y_scaler,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: LC Reconstruction ---
    print("\nApplying one shared LSST augmentation to the test set...")
    np.random.seed(args.lsst_seed)

    x_test_aug, x_test_retained_mask = apply_lsst_pipeline(
        x_test_clean,
        n_days,
        noise_std,
        samples_per_day=samples_per_day,
    )

    selected_samples = select_diagnostic_samples(
        x_test_clean,
        x_test_aug,
        args.sample_selections,
    )
    time_axis = np.arange(n_days) / samples_per_day

    for selection in selected_samples:
        sample_idx = selection["index"]
        selection_name = selection["name"]
        augmentation_rmse = selection["augmentation_rmse"]

        print(
            f"\nSelected {selection_name}: sample {sample_idx}, "
            f"augmentation RMSE = {augmentation_rmse:.6g}"
        )

        fig1, reconstructed_curve = plot_lc_reconstruction(
            sample_idx,
            x_test_clean,
            x_test_aug,
            x_test_retained_mask,
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

        out1 = output_dir / f"lc_reconstruction_{selection_name}.pdf"
        fig1.savefig(out1, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {out1}")

        fig3 = plot_augmentation_diagnostic(
            sample_idx,
            x_test_clean,
            x_test_aug,
            x_test_retained_mask,
            n_days,
            samples_per_day,
            augmentation_rmse,
        )

        out3 = (
            output_dir
            / f"augmentation_diagnostic_{selection_name}.pdf"
        )
        fig3.savefig(out3, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {out3}")
        sample_mask = x_test_retained_mask[sample_idx]

        out_data = (
            output_dir
            / f"light_curve_diagnostic_{selection_name}.csv"
        )

        save_sample_diagnostic_csv(
            out_data,
            time_axis,
            x_test_clean[sample_idx],
            x_test_aug[sample_idx],
            sample_mask,
            reconstructed_curve,
        )

        print(f"  Saved -> {out_data}")

        if not args.show:
            plt.close(fig1)
            plt.close(fig3)

    # --- Plot 2: Reconstruction Error ---
    print("\n[Plot 2] Computing per-timestep reconstruction errors...")
    fig2, error_curves = plot_reconstruction_error(
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
    out2 = output_dir / "reconstruction_error.pdf"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {out2}")
    error_csv = output_dir / "reconstruction_error.csv"
    save_reconstruction_error_csv(
        error_csv,
        time_axis,
        error_curves,
    )

    print(f"  Saved -> {error_csv}")

    if args.show:
        plt.show()
    else:
        plt.close(fig2)


if __name__ == "__main__":
    main()
