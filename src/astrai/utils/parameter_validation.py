"""Validation helpers for configured physical-parameter schemas."""

from collections.abc import Sequence


def validate_parameter_names(n_params, param_names):
    """Validate and normalise the configured parameter names.

    Parameters
    ----------
    n_params : int
        Expected number of model outputs.
    param_names : sequence of str
        Ordered names associated with those outputs.

    Returns
    -------
    tuple of str
        The validated names, preserving their configured order.
    """
    if (
        isinstance(n_params, bool)
        or not isinstance(n_params, int)
        or n_params <= 0
    ):
        raise ValueError("data.n_params must be a positive integer")
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
    return tuple(param_names)
