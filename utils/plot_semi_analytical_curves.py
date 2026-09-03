"""Plot clean curves from precomputed semi-analytical model datasets."""

import argparse
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re

import numpy as np
import yaml

from utils.data import load_raw_data


COLOURS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
DEFAULT_Y_LABEL = (
    r"$\log_{10}(L_{\rm bol}\,[{\rm erg}\,{\rm s}^{-1}])$"
)


def parse_parameter_selection(expression):
    """Parse ``NAME=VALUE`` pairs separated by commas."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Parameter selection must not be empty")

    selection = {}
    for item in expression.split(","):
        name, separator, raw_value = item.partition("=")
        name = name.strip()
        raw_value = raw_value.strip()
        if not separator or not name or not raw_value:
            raise ValueError(
                "Parameter selections must use NAME=VALUE pairs separated "
                f"by commas; received {expression!r}"
            )
        if name in selection:
            raise ValueError(
                f"Parameter {name!r} occurs more than once in {expression!r}"
            )
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"Value for parameter {name!r} is not numeric: "
                f"{raw_value!r}"
            ) from error
        if not np.isfinite(value):
            raise ValueError(
                f"Value for parameter {name!r} must be finite"
            )
        selection[name] = value

    return selection


def find_unique_sample(
    parameters,
    param_names,
    selection,
    *,
    rtol=1e-7,
    atol=1e-8,
):
    """Return the unique row matching an exact parameter selection."""
    parameters = np.asarray(parameters)
    if parameters.ndim != 2:
        raise ValueError("Parameter values must be a two-dimensional array")
    if parameters.shape[1] != len(param_names):
        raise ValueError(
            "Parameter array width does not match parameter names: "
            f"{parameters.shape[1]} != {len(param_names)}"
        )
    if not isinstance(selection, Mapping) or not selection:
        raise ValueError("At least one parameter value must be selected")

    name_to_column = {name: index for index, name in enumerate(param_names)}
    unknown = [name for name in selection if name not in name_to_column]
    if unknown:
        raise ValueError(
            "Unknown parameter name(s): " + ", ".join(sorted(unknown))
        )

    mask = np.ones(len(parameters), dtype=bool)
    for name, expected in selection.items():
        if isinstance(expected, bool) or not isinstance(
            expected,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(f"Value for parameter {name!r} must be numeric")
        if not np.isfinite(expected):
            raise ValueError(f"Value for parameter {name!r} must be finite")
        mask &= np.isclose(
            parameters[:, name_to_column[name]],
            expected,
            rtol=rtol,
            atol=atol,
        )

    indices = np.flatnonzero(mask)
    if len(indices) != 1:
        rendered = ", ".join(
            f"{name}={value:g}" for name, value in selection.items()
        )
        raise ValueError(
            f"Expected exactly one sample for [{rendered}], "
            f"found {len(indices)}"
        )
    return int(indices[0])


def validate_sample_index(index, n_samples):
    """Validate and return one zero-based sample index."""
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError(f"Sample index must be an integer; received {index!r}")
    if index < 0 or index >= n_samples:
        raise ValueError(
            f"Sample index {index} is outside the valid range "
            f"0..{n_samples - 1}"
        )
    return int(index)


def resolve_selected_indices(
    parameters,
    param_names,
    indices=None,
    selections=None,
):
    """Resolve explicit row indices and parameter selections in order."""
    resolved = []
    for index in indices or ():
        resolved.append(validate_sample_index(index, len(parameters)))
    for selection in selections or ():
        resolved.append(
            find_unique_sample(parameters, param_names, selection)
        )

    unique = []
    seen = set()
    for index in resolved:
        if index not in seen:
            unique.append(index)
            seen.add(index)
    return unique


def _numeric_levels(parameter, values):
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
    ):
        raise ValueError(
            f"Levels for parameter {parameter!r} must be a non-empty sequence"
        )

    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(
                f"Level for parameter {parameter!r} must be numeric"
            )
        if not np.isfinite(value):
            raise ValueError(
                f"Level for parameter {parameter!r} must be finite"
            )
        result.append(float(value))
    return result


def resolve_one_at_a_time_groups(parameters, param_names, specification):
    """Resolve configured one-at-a-time levels to exact dataset rows."""
    if not isinstance(specification, Mapping):
        raise ValueError("one_at_a_time must be a mapping")

    reference = specification.get("reference")
    levels = specification.get("levels")
    if not isinstance(reference, Mapping):
        raise ValueError("one_at_a_time.reference must be a mapping")
    if not isinstance(levels, Mapping) or not levels:
        raise ValueError(
            "one_at_a_time.levels must be a non-empty mapping"
        )

    missing_reference = [
        name for name in param_names if name not in reference
    ]
    extra_reference = [
        name for name in reference if name not in param_names
    ]
    if missing_reference or extra_reference:
        details = []
        if missing_reference:
            details.append("missing " + ", ".join(missing_reference))
        if extra_reference:
            details.append("unknown " + ", ".join(extra_reference))
        raise ValueError(
            "one_at_a_time.reference must define every parameter ("
            + "; ".join(details)
            + ")"
        )

    unknown_levels = [name for name in levels if name not in param_names]
    if unknown_levels:
        raise ValueError(
            "Unknown parameter name(s) in one_at_a_time.levels: "
            + ", ".join(unknown_levels)
        )

    titles = specification.get("titles", {})
    if not isinstance(titles, Mapping):
        raise ValueError("one_at_a_time.titles must be a mapping")

    groups = []
    for parameter in param_names:
        if parameter not in levels:
            continue
        rows = []
        for value in _numeric_levels(parameter, levels[parameter]):
            selection = dict(reference)
            selection[parameter] = value
            rows.append(
                (value, find_unique_sample(parameters, param_names, selection))
            )
        groups.append(
            {
                "parameter": parameter,
                "title": titles.get(parameter, parameter),
                "rows": rows,
            }
        )
    return groups


def resolve_quantile_summary_groups(
    curves,
    parameters,
    param_names,
    specification,
):
    """Build marginal median curves for configured parameter quantiles."""
    if not isinstance(specification, Mapping):
        raise ValueError("quantile_summary must be a mapping")
    quantile_ranges = specification.get("quantile_ranges")
    if not isinstance(quantile_ranges, Mapping) or not quantile_ranges:
        raise ValueError(
            "quantile_summary.quantile_ranges must be a non-empty mapping"
        )

    curves = np.asarray(curves)
    parameters = np.asarray(parameters)
    if curves.ndim != 2 or parameters.ndim != 2:
        raise ValueError("Curves and parameters must be two-dimensional")
    if len(curves) != len(parameters):
        raise ValueError(
            "Curves and parameters contain different sample counts"
        )
    if parameters.shape[1] != len(param_names):
        raise ValueError(
            "Parameter array width does not match parameter names: "
            f"{parameters.shape[1]} != {len(param_names)}"
        )

    ranges = []
    for label, bounds in quantile_ranges.items():
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(bounds, Sequence)
            or isinstance(bounds, (str, bytes))
            or len(bounds) != 2
        ):
            raise ValueError(
                "Each quantile range must map a label to [lower, upper]"
            )
        lower, upper = bounds
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower < 0
            or upper > 1
            or lower >= upper
        ):
            raise ValueError(
                f"Quantile range {label!r} must satisfy "
                "0 <= lower < upper <= 1"
            )
        ranges.append((label, float(lower), float(upper)))

    titles = specification.get("titles", {})
    if not isinstance(titles, Mapping):
        raise ValueError("quantile_summary.titles must be a mapping")

    n_samples = len(parameters)
    groups = []
    for column, parameter in enumerate(param_names):
        order = np.argsort(parameters[:, column], kind="stable")
        summaries = []
        for label, lower, upper in ranges:
            start = math.floor(lower * n_samples)
            stop = math.ceil(upper * n_samples)
            if upper == 1:
                stop = n_samples
            selected = order[start:stop]
            if len(selected) == 0:
                raise ValueError(
                    f"Quantile range {label!r} selects no samples"
                )

            selected_values = parameters[selected, column]
            lower_value = float(np.min(selected_values))
            upper_value = float(np.max(selected_values))
            summaries.append(
                {
                    "label": (
                        f"{label}: {lower_value:g}-{upper_value:g} "
                        f"(n={len(selected)})"
                    ),
                    "curve": np.median(curves[selected], axis=0),
                }
            )
            print(
                f"{parameter} / {label}: n={len(selected)}, "
                f"range={lower_value:g}..{upper_value:g}"
            )

        groups.append(
            {
                "parameter": parameter,
                "title": titles.get(parameter, parameter),
                "summaries": summaries,
            }
        )
    return groups


def build_time_axis(n_days, samples_per_day):
    """Build a time axis in days from the configured sampling rate."""
    if isinstance(samples_per_day, bool) or not isinstance(
        samples_per_day,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("data.samples_per_day must be numeric")
    if not np.isfinite(samples_per_day) or samples_per_day <= 0:
        raise ValueError("data.samples_per_day must be greater than zero")
    return np.arange(n_days, dtype=np.float64) / float(samples_per_day)


def _pyplot(show):
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_figure(
    figure,
    output_dir,
    output_name,
    formats,
    *,
    dpi,
    show,
):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output_name):
        raise ValueError(
            "Output name may contain only letters, numbers, '.', '_' and '-'"
        )
    if not formats:
        raise ValueError("At least one output format must be selected")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for output_format in dict.fromkeys(formats):
        if output_format not in {"pdf", "png"}:
            raise ValueError(
                f"Unsupported output format {output_format!r}"
            )
        path = output_dir / f"{output_name}.{output_format}"
        options = {"bbox_inches": "tight"}
        if output_format == "png":
            options["dpi"] = dpi
        figure.savefig(path, **options)
        saved.append(path)

    plt = _pyplot(show)
    if show:
        plt.show()
    plt.close(figure)
    return saved


def plot_selected_curves(
    curves,
    parameters,
    param_names,
    indices,
    time,
    output_dir,
    output_name,
    formats,
    *,
    y_label=DEFAULT_Y_LABEL,
    dpi=300,
    show=False,
):
    """Plot explicitly selected, existing dataset rows."""
    if not indices:
        raise ValueError("At least one sample must be selected")
    plt = _pyplot(show)
    figure, axis = plt.subplots(figsize=(10.5, 5.8))

    for position, index in enumerate(indices):
        index = validate_sample_index(index, len(curves))
        axis.plot(
            time,
            curves[index],
            color=COLOURS[position % len(COLOURS)],
            linewidth=1.8,
            label=f"Sample {index}",
        )
        values = ", ".join(
            f"{name}={value:g}"
            for name, value in zip(param_names, parameters[index])
        )
        print(f"Sample {index}: {values}")

    axis.set_title("Clean semi-analytical light curves")
    axis.set_xlabel("Days from explosion")
    axis.set_ylabel(y_label)
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.grid(True, alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    return _save_figure(
        figure,
        output_dir,
        output_name,
        formats,
        dpi=dpi,
        show=show,
    )


def plot_one_at_a_time(
    curves,
    groups,
    time,
    output_dir,
    output_name,
    formats,
    *,
    y_label=DEFAULT_Y_LABEL,
    dpi=300,
    show=False,
):
    """Plot exact one-at-a-time comparisons defined by configuration."""
    if not groups:
        raise ValueError("No one-at-a-time parameter groups were configured")

    plt = _pyplot(show)
    n_columns = min(3, math.ceil(math.sqrt(len(groups))))
    n_rows = math.ceil(len(groups) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.4 * n_columns, 3.7 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    flat_axes = axes.ravel()
    for axis, group in zip(flat_axes, groups):
        for position, (value, index) in enumerate(group["rows"]):
            axis.plot(
                time,
                curves[index],
                color=COLOURS[position % len(COLOURS)],
                linewidth=1.9,
                label=f"{value:g}",
            )
        axis.set_title(group["title"])
        axis.set_xlim(float(time[0]), float(time[-1]))
        axis.grid(True, alpha=0.22)
        axis.legend(title=group["parameter"], frameon=False)

    for axis in flat_axes[len(groups):]:
        axis.set_visible(False)

    figure.supxlabel("Days from explosion", y=0.025)
    figure.supylabel(y_label, x=0.015)
    figure.tight_layout(rect=(0.04, 0.06, 1.0, 0.99))
    return _save_figure(
        figure,
        output_dir,
        output_name,
        formats,
        dpi=dpi,
        show=show,
    )


def plot_quantile_summary(
    groups,
    time,
    output_dir,
    output_name,
    formats,
    *,
    y_label=DEFAULT_Y_LABEL,
    dpi=300,
    show=False,
):
    """Plot marginal median curves for each configured parameter."""
    if not groups:
        raise ValueError("No quantile-summary parameter groups were built")

    plt = _pyplot(show)
    n_columns = min(3, math.ceil(math.sqrt(len(groups))))
    n_rows = math.ceil(len(groups) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.4 * n_columns, 3.7 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    flat_axes = axes.ravel()
    for axis, group in zip(flat_axes, groups):
        for position, summary in enumerate(group["summaries"]):
            axis.plot(
                time,
                summary["curve"],
                color=COLOURS[position % len(COLOURS)],
                linewidth=1.9,
                label=summary["label"],
            )
        axis.set_title(group["title"])
        axis.set_xlim(float(time[0]), float(time[-1]))
        axis.grid(True, alpha=0.22)
        axis.legend(frameon=False, fontsize=8)

    for axis in flat_axes[len(groups):]:
        axis.set_visible(False)

    figure.suptitle(
        "Median clean light curves by parameter quantile",
        fontsize=15,
        y=0.995,
    )
    figure.supxlabel("Days from explosion", y=0.025)
    figure.supylabel(y_label, x=0.015)
    figure.tight_layout(rect=(0.04, 0.06, 1.0, 0.96))
    return _save_figure(
        figure,
        output_dir,
        output_name,
        formats,
        dpi=dpi,
        show=show,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot clean curves already present in a configured "
            "semi-analytical dataset."
        )
    )
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument(
        "--data-path",
        "--data_path",
        dest="data_path",
        help="Optional override for a Parquet dataset path",
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        help=(
            "Base directory for relative data paths "
            "(default: repository root)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        required=True,
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--output-name",
        "--output_name",
        dest="output_name",
        help="Output filename without extension",
    )
    parser.add_argument(
        "--index",
        action="append",
        type=int,
        default=[],
        help="Zero-based dataset row to plot; may be repeated",
    )
    parser.add_argument(
        "--parameters",
        action="append",
        default=[],
        metavar="NAME=VALUE,...",
        help=(
            "Exact parameter selection; may be repeated. A selection must "
            "match exactly one dataset row."
        ),
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=("pdf", "png"),
        dest="formats",
        help="Output format; may be repeated (default: PDF and PNG)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive plot window",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero")

    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    repository_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root) if args.data_root else repository_root
    curves, parameters = load_raw_data(
        args.data_path,
        cfg,
        data_root=data_root,
    )
    if parameters is None:
        raise ValueError(
            "The configured dataset does not contain parameter columns"
        )

    data_cfg = cfg["data"]
    param_names = list(data_cfg["param_names"])
    time = build_time_axis(
        curves.shape[1],
        data_cfg.get("samples_per_day", 1),
    )
    formats = args.formats or ["pdf", "png"]
    visualisation = cfg.get("visualisation", {})
    if not isinstance(visualisation, Mapping):
        raise ValueError("visualisation must be a mapping")
    y_label = visualisation.get("y_label", DEFAULT_Y_LABEL)

    print(
        f"Loaded {len(curves)} curves with {curves.shape[1]} samples and "
        f"{parameters.shape[1]} parameters."
    )

    if args.index or args.parameters:
        selections = [
            parse_parameter_selection(expression)
            for expression in args.parameters
        ]
        indices = resolve_selected_indices(
            parameters,
            param_names,
            args.index,
            selections,
        )
        output_name = args.output_name or (
            f"clean_light_curves_{len(param_names)}par_selected"
        )
        saved = plot_selected_curves(
            curves,
            parameters,
            param_names,
            indices,
            time,
            args.output_dir,
            output_name,
            formats,
            y_label=y_label,
            dpi=args.dpi,
            show=args.show,
        )
    else:
        one_at_a_time = visualisation.get("one_at_a_time")
        quantile_summary = visualisation.get("quantile_summary")
        if one_at_a_time is not None and quantile_summary is not None:
            raise ValueError(
                "Configure only one default visualisation: "
                "one_at_a_time or quantile_summary"
            )
        if one_at_a_time is not None:
            groups = resolve_one_at_a_time_groups(
                parameters,
                param_names,
                one_at_a_time,
            )
            output_name = args.output_name or one_at_a_time.get(
                "output_name",
                f"clean_light_curves_{len(param_names)}par",
            )
            saved = plot_one_at_a_time(
                curves,
                groups,
                time,
                args.output_dir,
                output_name,
                formats,
                y_label=y_label,
                dpi=args.dpi,
                show=args.show,
            )
        elif quantile_summary is not None:
            groups = resolve_quantile_summary_groups(
                curves,
                parameters,
                param_names,
                quantile_summary,
            )
            output_name = args.output_name or quantile_summary.get(
                "output_name",
                f"clean_light_curves_{len(param_names)}par_quantiles",
            )
            saved = plot_quantile_summary(
                groups,
                time,
                args.output_dir,
                output_name,
                formats,
                y_label=y_label,
                dpi=args.dpi,
                show=args.show,
            )
        else:
            raise ValueError(
                "No samples were selected and the configuration has no "
                "default visualisation specification. Use --index or "
                "--parameters."
            )

    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
