"""Collect reproducibility-relevant details about the runtime environment.

The collected values are suitable for experiment metadata: they describe the
Python and operating-system environment, installed distribution versions and
the effective PyTorch execution state without exposing hostnames or local
installation paths. Collection is observational and must not consume random
numbers or alter deterministic settings.
"""

import os
import platform
import re
from importlib import metadata as importlib_metadata

import torch


RUNTIME_ENVIRONMENT_VERSION = 1

_RELEVANT_ENVIRONMENT_VARIABLES = (
    "CUBLAS_WORKSPACE_CONFIG",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
)


def _normalise_distribution_name(name):
    """Return the canonical spelling used to key package versions."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_distribution_versions():
    """Return installed distribution names and versions without local paths."""
    versions = {}
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name or version is None:
            continue
        versions[_normalise_distribution_name(name)] = str(version)
    return dict(sorted(versions.items()))


def _mps_metadata():
    """Describe the optional Apple Metal backend without assuming it exists."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return {"built": False, "available": False}
    return {
        "built": bool(backend.is_built()),
        "available": bool(backend.is_available()),
    }


def _selected_device_metadata(device, cuda_available):
    """Describe the selected torch device, avoiding unnecessary CUDA access."""
    if device is None:
        return None

    selected = torch.device(device)
    result = {
        "value": str(selected),
        "type": selected.type,
        "index": selected.index,
    }
    if selected.type == "cuda" and cuda_available:
        index = selected.index
        if index is None:
            index = torch.cuda.current_device()
        result.update(
            {
                "index": int(index),
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return result


def capture_torch_environment(device=None):
    """Return the effective PyTorch backend and execution state."""
    cuda_available = bool(torch.cuda.is_available())
    cudnn_available = bool(torch.backends.cudnn.is_available())
    warn_only_getter = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        None,
    )

    return {
        "version": str(torch.__version__),
        "selected_device": _selected_device_metadata(
            device,
            cuda_available,
        ),
        "cuda": {
            "available": cuda_available,
            "build_version": torch.version.cuda,
        },
        "cudnn": {
            "available": cudnn_available,
            "version": (
                torch.backends.cudnn.version() if cudnn_available else None
            ),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic": bool(torch.backends.cudnn.deterministic),
        },
        "mps": _mps_metadata(),
        "threads": {
            "intra_op": int(torch.get_num_threads()),
            "inter_op": int(torch.get_num_interop_threads()),
        },
        "deterministic_algorithms": {
            "enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "warn_only": (
                None if warn_only_getter is None else bool(warn_only_getter())
            ),
        },
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def capture_execution_environment(device=None):
    """Return runtime settings that may change during the current process."""
    return {
        "pytorch": capture_torch_environment(device=device),
        "environment_variables": {
            name: os.environ[name]
            for name in _RELEVANT_ENVIRONMENT_VARIABLES
            if name in os.environ
        },
    }


def capture_runtime_environment(device=None):
    """Return a metadata-safe snapshot of the current execution environment."""
    snapshot = {
        "schema_version": RUNTIME_ENVIRONMENT_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "installed_distributions": installed_distribution_versions(),
    }
    snapshot.update(capture_execution_environment(device=device))
    return snapshot
