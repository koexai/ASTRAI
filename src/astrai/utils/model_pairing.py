"""Verify the shared representation before composing independently trained stages."""

from pathlib import Path

import yaml

from astrai.utils.partitions import Partition
from astrai.utils.preprocessing import (
    BUNDLE_SCHEMA_VERSION,
    FIT_POLICY,
    _digest,
    _learned_state_digest,
)
from astrai.utils.target_transformations import validate_parameter_scalers


def _selected_bundle(exp_dir, stage, scaler):
    """Read the selected holdout bundle, including its scaler association."""
    path = Path(exp_dir) / "metadata.yaml"
    try:
        metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
        version = metadata.get("experiment_metadata_version")
        if type(version) is not int or version < 5:
            raise ValueError("Experiment metadata lacks checkpoint-to-bundle provenance")
        if metadata["run"]["stage"] != stage or metadata["run"]["status"] != "completed":
            raise ValueError(f"Expected a completed {stage} experiment")
        selected = metadata["checkpoint"]["selected_checkpoint"]
        fold = selected["outer_fold"]
        if type(fold) is not int or fold < 1 or selected["role"] != "holdout_validation":
            raise ValueError("Expected a selected outer-fold holdout checkpoint")
        manifests = [
            item["preprocessing"] for item in metadata["results"]["folds"]
            if item["outer_fold"] == fold
        ]
        if len(manifests) != 1:
            raise ValueError("Selected checkpoint must identify exactly one preprocessing bundle")
        manifest = manifests[0]
        identity = dict(manifest)
        bundle_id = identity.pop("bundle_id")
        if (manifest["bundle_schema_version"] != BUNDLE_SCHEMA_VERSION
                or manifest["fit_policy"] != FIT_POLICY
                or not manifest["dataset_id"]
                or bundle_id != _digest(identity)
                or bundle_id != selected["preprocessing_bundle_id"]):
            raise ValueError("Invalid selected preprocessing identity")
        partition = Partition.from_record(manifest["partition"])
        if partition.outer_fold != fold or partition.role != selected["role"]:
            raise ValueError("Selected checkpoint and preprocessing fold or role differ")
        association = getattr(scaler, "astrai_association", {})
        for key in ("bundle_id", "dataset_id", "fit_policy", "target_transform",
                    "param_names", "parameter_units", "unit_convention"):
            if key not in association or association[key] != manifest[key]:
                raise ValueError(f"Parameter scaler association differs: {key}")
        if _learned_state_digest(scaler) != manifest["learned_state"]["y_scaler"]:
            raise ValueError("Parameter scaler state differs from the selected bundle")
        return manifest
    except (OSError, KeyError, TypeError, AttributeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Cannot verify cascade pairing for {stage} at {path}: {exc}. "
            "Use completed split checkpoints with recorded shared preprocessing; "
            "historical pairs without this provenance cannot be composed."
        ) from exc


def validate_cascade_pair(exp_char, exp_gen, char_y_scaler, gen_y_scaler):
    """Require the same fold, pool and bundle; selected epochs may differ.

    Call after the checkpoint loaders, which verify model and preprocessing
    file hashes. Matching numerical scaler values alone do not prove pairing.
    No historical artefacts are converted or associated implicitly.
    """
    char_bundle = _selected_bundle(exp_char, "characterizer", char_y_scaler)
    gen_bundle = _selected_bundle(exp_gen, "generator", gen_y_scaler)
    if char_bundle["bundle_id"] != gen_bundle["bundle_id"]:
        raise ValueError(
            "Incompatible cascade checkpoints: PPReg and LCGen must use the same "
            "outer fold, dataset, training pool and preprocessing bundle. "
            "Choose a verified pair trained on the same shared fold; no automatic "
            "conversion or cross-fold association is supported."
        )
    validate_parameter_scalers(
        char_y_scaler, gen_y_scaler,
        context="characterizer output and generator input scaling",
    )
