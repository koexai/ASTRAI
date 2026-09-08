"""Canonical transformations between ASTRAI target representation spaces."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml


TARGET_TRANSFORM_NAME = "log1p"
TARGET_TRANSFORM_VERSION = 1
PREPROCESSING_ARTEFACT_SCHEMA_VERSION = 5


def target_transform_contract(cfg=None):
    """Return and validate the target-transformation contract for *cfg*.

    Configurations created before the contract was made explicit are treated
    as ``log1p`` configurations. No alternative transform is currently
    supported.
    """
    cfg = {} if cfg is None else cfg
    if not isinstance(cfg, Mapping):
        raise ValueError("Configuration must be a mapping")

    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ValueError("Configuration data section must be a mapping")

    transform = data_cfg.get("target_transform", TARGET_TRANSFORM_NAME)
    if transform != TARGET_TRANSFORM_NAME:
        raise ValueError(
            f"Unsupported data.target_transform {transform!r}; "
            f"expected {TARGET_TRANSFORM_NAME!r}"
        )

    return {
        "name": TARGET_TRANSFORM_NAME,
        "version": TARGET_TRANSFORM_VERSION,
        "input_space": "physical",
        "model_space": "transformed",
        "physical_domain": "non_negative",
    }


def validate_physical_targets(values, *, context="Physical target values"):
    """Return *values* as an array after validating the physical domain."""
    array = np.asarray(values)
    if not np.isfinite(array).all():
        raise ValueError(f"{context} must contain only finite values")
    if np.any(array < 0):
        raise ValueError(f"{context} must be non-negative")
    return array


def physical_to_transformed(values, cfg=None):
    """Map finite, non-negative physical parameters to model target space."""
    target_transform_contract(cfg)
    physical = validate_physical_targets(values)
    return np.log1p(physical)


def transformed_to_physical(values, cfg=None):
    """Map model-space parameters to physical space without clipping."""
    target_transform_contract(cfg)
    transformed = np.asarray(values)
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed target values must contain only finite values")
    return np.expm1(transformed)


def physical_to_scaled(values, scaler, cfg=None):
    """Map physical parameters through the target transform and scaler."""
    return scaler.transform(physical_to_transformed(values, cfg))


def scaled_to_transformed(values, scaler):
    """Undo target scaling while retaining transformed target space."""
    scaled = np.asarray(values)
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled target values must contain only finite values")
    return scaler.inverse_transform(scaled)


def scaled_to_physical(values, scaler, cfg=None):
    """Decode scaled model targets into physical parameters."""
    return transformed_to_physical(scaled_to_transformed(values, scaler), cfg)


def validate_target_contract_compatibility(left, right, *, context="targets"):
    """Reject target contracts that cannot share scaled model values."""
    expected_keys = (
        "name",
        "version",
        "input_space",
        "model_space",
        "physical_domain",
    )
    left_values = {key: left.get(key) for key in expected_keys}
    right_values = {key: right.get(key) for key in expected_keys}
    if left_values != right_values:
        raise ValueError(
            f"Incompatible target transformation contracts for {context}: "
            f"{left_values!r} != {right_values!r}"
        )


def validate_parameter_scalers(left, right, *, context="parameter scaling"):
    """Ensure two model stages use identical target scaling."""
    if type(left) is not type(right):
        raise ValueError(f"Incompatible {context}: scaler types differ")

    for attribute in ("mean_", "scale_"):
        if not hasattr(left, attribute) or not hasattr(right, attribute):
            raise ValueError(
                f"Incompatible {context}: both scalers must expose {attribute}"
            )
        left_value = np.asarray(getattr(left, attribute), dtype=float)
        right_value = np.asarray(getattr(right, attribute), dtype=float)
        if (
            left_value.shape != right_value.shape
            or not np.allclose(
                left_value,
                right_value,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
        ):
            raise ValueError(
                f"Incompatible {context}: {attribute} values differ"
            )


def _normalise_recorded_contract(contract, *, source):
    if not isinstance(contract, Mapping):
        raise ValueError(
            f"Target transformation metadata in {source} must be a mapping"
        )
    expected = target_transform_contract()
    validate_target_contract_compatibility(
        expected,
        contract,
        context=str(source),
    )
    return dict(expected)


def validate_preprocessing_target_contract(preprocessing_dir, cfg):
    """Validate target metadata for a preprocessing run.

    Metadata-free directories remain supported as legacy artefacts. Versioned
    preprocessing runs created before schema 5 are intentionally rejected
    because their target representation is not self-describing.
    """
    source_dir = Path(preprocessing_dir).expanduser().resolve()
    metadata_path = source_dir / "metadata.yaml"
    expected = target_transform_contract(cfg)
    if not metadata_path.is_file():
        return expected

    with metadata_path.open(encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    status = metadata.get("run", {}).get("status")
    if status != "completed":
        raise ValueError(
            "Preprocessing metadata must describe a completed run; "
            f"found status {status!r} in {metadata_path}"
        )
    schema = metadata.get("preprocessing_artefact_schema_version")
    if schema != PREPROCESSING_ARTEFACT_SCHEMA_VERSION:
        raise ValueError(
            "Preprocessing target representation is not compatible with "
            f"schema {PREPROCESSING_ARTEFACT_SCHEMA_VERSION}; found schema "
            f"{schema!r} in {metadata_path}. Regenerate preprocessing artefacts."
        )
    recorded = _normalise_recorded_contract(
        metadata.get("target_transform"),
        source=metadata_path,
    )
    validate_target_contract_compatibility(
        expected,
        recorded,
        context="configuration and preprocessing run",
    )
    return recorded


def experiment_target_contract(exp_dir, cfg):
    """Resolve a model experiment's target contract with legacy fallback."""
    expected = target_transform_contract(cfg)
    if exp_dir is None:
        return expected

    experiment_dir = Path(exp_dir).expanduser().resolve()
    metadata_path = experiment_dir / "metadata.yaml"
    if not metadata_path.is_file():
        return expected

    with metadata_path.open(encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    version = metadata.get("experiment_metadata_version")
    recorded_data = metadata.get("data", {})
    configured_data = cfg.get("data", {})
    for key in ("n_params", "param_names"):
        recorded_value = recorded_data.get(key)
        configured_value = configured_data.get(key)
        if (
            recorded_value is not None
            and configured_value is not None
            and recorded_value != configured_value
        ):
            raise ValueError(
                f"Experiment {key} in {metadata_path} is incompatible with "
                f"the configuration: {recorded_value!r} != {configured_value!r}"
            )

    recorded = recorded_data.get("target_transform")
    if recorded is None:
        if isinstance(version, int) and version >= 3:
            raise ValueError(
                f"Experiment metadata version {version} in {metadata_path} "
                "does not record a target transformation contract"
            )
        return expected

    recorded = _normalise_recorded_contract(recorded, source=metadata_path)
    validate_target_contract_compatibility(
        expected,
        recorded,
        context="configuration and experiment",
    )
    return recorded


def load_fold_target_array(fold_dir, canonical_name, legacy_name=None):
    """Load a canonical fold target, optionally falling back for legacy runs."""
    fold_dir = Path(fold_dir)
    canonical_path = fold_dir / canonical_name
    if canonical_path.is_file():
        return np.load(canonical_path, allow_pickle=False)
    if legacy_name is not None:
        legacy_path = fold_dir / legacy_name
        if legacy_path.is_file():
            return np.load(legacy_path, allow_pickle=False)
    raise FileNotFoundError(f"Missing preprocessing artefact: {canonical_path}")
