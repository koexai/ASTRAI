"""Data loading utilities shared by training and analysis entry points."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_DATA_FORMATS = {"npy_csv", "parquet"}


def _positive_int(data_cfg, key):
    value = data_cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"data.{key} must be a positive integer")
    return value


def _data_contract(cfg):
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("Configuration must contain a 'data' mapping")

    data_format = data_cfg.get("format", "parquet")
    if data_format not in SUPPORTED_DATA_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_DATA_FORMATS))
        raise ValueError(
            f"Unsupported data format {data_format!r}; expected one of: "
            f"{supported}"
        )

    n_days = _positive_int(data_cfg, "n_days")
    n_params = _positive_int(data_cfg, "n_params")
    param_names = data_cfg.get("param_names")

    if (
        not isinstance(param_names, Sequence)
        or isinstance(param_names, (str, bytes))
        or not param_names
        or any(not isinstance(name, str) or not name for name in param_names)
    ):
        raise ValueError(
            "data.param_names must be a non-empty sequence of names"
        )
    if len(set(param_names)) != len(param_names):
        raise ValueError("data.param_names must not contain duplicates")
    if len(param_names) != n_params:
        raise ValueError(
            "data.n_params does not match data.param_names: "
            f"{n_params} != {len(param_names)}"
        )

    return data_cfg, data_format, n_days, list(param_names)


def _resolve_path(path, data_root):
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(data_root) / resolved
    return resolved


def _parameter_values(frame, param_names, source):
    present = [name in frame.columns for name in param_names]
    if not any(present):
        return None
    if not all(present):
        missing = [
            name for name, is_present in zip(param_names, present)
            if not is_present
        ]
        raise ValueError(
            f"Parameter data in {source} are incomplete; missing columns: "
            f"{', '.join(missing)}"
        )
    return frame[param_names].to_numpy(dtype=np.float32)


def _validate_loaded_arrays(curves, parameters, n_days, source):
    if curves.ndim != 2:
        raise ValueError(
            f"Curves loaded from {source} must be a two-dimensional array; "
            f"found shape {curves.shape}"
        )
    if curves.shape[1] != n_days:
        raise ValueError(
            f"Curves loaded from {source} have {curves.shape[1]} samples, "
            f"but data.n_days is {n_days}"
        )
    if parameters is not None and len(parameters) != len(curves):
        raise ValueError(
            f"Curves and parameters loaded from {source} contain different "
            f"sample counts: {len(curves)} and {len(parameters)}"
        )


def load_raw_data(data_path, cfg, data_root=None):
    """Load curves and optional physical parameters without transformations.

    ``npy_csv`` datasets use ``data.curves_path`` and ``data.params_path``
    from the configuration. ``parquet`` datasets use *data_path* when given,
    otherwise ``data.path``. Relative paths are resolved from *data_root*, or
    from the current working directory when it is omitted.

    The returned arrays are ``float32`` to preserve the existing loader's
    numerical contract. Parameters remain in their original physical space;
    callers that require a transformation must apply it explicitly.
    """
    data_cfg, data_format, n_days, param_names = _data_contract(cfg)
    root = Path.cwd() if data_root is None else Path(data_root)

    if data_format == "npy_csv":
        try:
            curves_path = data_cfg["curves_path"]
            params_path = data_cfg["params_path"]
        except KeyError as error:
            raise ValueError(
                f"data.{error.args[0]} is required for npy_csv datasets"
            ) from error

        curves_file = _resolve_path(curves_path, root)
        params_file = _resolve_path(params_path, root)
        curves = np.asarray(np.load(curves_file), dtype=np.float32)
        separator = data_cfg.get("params_csv_sep", ",")
        parameter_frame = pd.read_csv(params_file, sep=separator)
        parameters = _parameter_values(
            parameter_frame,
            param_names,
            params_file,
        )
        source = curves_file
    else:
        configured_path = data_path or data_cfg.get("path")
        if configured_path is None:
            raise ValueError(
                "A Parquet path is required via data.path or data_path"
            )
        parquet_file = _resolve_path(configured_path, root)
        frame = pd.read_parquet(parquet_file)
        curve_columns = [str(index) for index in range(n_days)]
        missing_curve_columns = [
            name for name in curve_columns if name not in frame.columns
        ]
        if missing_curve_columns:
            preview = ", ".join(missing_curve_columns[:5])
            if len(missing_curve_columns) > 5:
                preview += ", ..."
            raise ValueError(
                f"Curve data in {parquet_file} are incomplete; missing "
                f"columns: {preview}"
            )
        curves = frame[curve_columns].to_numpy(dtype=np.float32)
        parameters = _parameter_values(frame, param_names, parquet_file)
        source = parquet_file

    _validate_loaded_arrays(curves, parameters, n_days, source)
    return curves, parameters
