"""Pool-owned fit/transform operations and verifiable preprocessing bundles.

Anti-leakage restricts learned state to the training rows of a Partition.
ASTRAI additionally fits curve scaling/PCA on clean representations only;
that policy is specific to this application, not a general CV requirement.
"""
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yaml

from astrai.utils.array_dtypes import as_model_array
from astrai.utils.masking import view_configuration
from astrai.utils.partitions import Partition
from astrai.utils.target_transformations import (
    PREPROCESSING_ARTEFACT_SCHEMA_VERSION, target_transform_contract,
)


BUNDLE_SCHEMA_VERSION = 1
FIT_POLICY = "clean_original_samples"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def array_digest(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def dataset_identity(curves, transformed_parameters):
    """Identify canonical ordered data; a digest never replaces recoverable data."""
    return _digest([array_digest(as_model_array(curves)),
                    array_digest(as_model_array(transformed_parameters))])


def validate_pca_size(n_components, n_training, n_features):
    if isinstance(n_components, bool) or not isinstance(n_components, Integral):
        raise ValueError("PCA components must be a positive integer")
    if n_training < 2 or not 1 <= n_components <= min(n_training, n_features):
        raise ValueError(f"PCA components={n_components} are not admissible for "
                         f"{n_training} clean training rows and {n_features} features")


def parameter_semantics(cfg, n_parameters):
    data = cfg.get("data", {})
    names = data.get("param_names")
    if names is not None and (len(names) != n_parameters or len(set(names)) != len(names)):
        raise ValueError("Parameter names must match the canonical parameter columns")
    if data.get("n_params", n_parameters) != n_parameters:
        raise ValueError("Parameter count differs from the dataset")
    units = data.get("param_units")
    if units is not None and (not isinstance(units, (list, tuple))
                              or len(units) != n_parameters
                              or any(not isinstance(unit, str) or not unit for unit in units)):
        raise ValueError("Explicit parameter units must match the parameter columns")
    return {"target_transform": target_transform_contract(cfg),
            "n_params": n_parameters, "param_names": names,
            "parameter_units": None if units is None else list(units),
            "unit_convention": "source_dataset"}


def _learned_state_digest(obj):
    attributes = {name: array_digest(value) for name, value in vars(obj).items()
                  if name.endswith("_") and isinstance(value, (np.ndarray, np.number, int, float))}
    return _digest({"type": type(obj).__name__, "state": attributes,
                    "parameters": obj.get_params()})


@dataclass
class PreprocessingBundle:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    pca: PCA
    manifest: dict

    def transform_curves(self, clean_or_augmented):
        """Transform any permitted view without changing learned state."""
        return as_model_array(self.pca.transform(self.x_scaler.transform(clean_or_augmented)))

    def transform_parameters(self, transformed):
        return as_model_array(self.y_scaler.transform(transformed))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if any((directory / name).exists() for name in ("bundle.yaml", "x_scaler.pkl", "y_scaler.pkl", "pca.pkl")):
            raise FileExistsError(f"Preprocessing bundle already exists: {directory}")
        for name in ("x_scaler", "y_scaler", "pca"):
            joblib.dump(getattr(self, name), directory / f"{name}.pkl")
        (directory / "bundle.yaml").write_text(yaml.safe_dump(self.manifest, sort_keys=False), encoding="utf-8")


def fit_preprocessing(curves, transformed_parameters, partition, n_components,
                      pca_seed, cfg=None, *, data_id=None):
    """Create fresh learned objects on the explicitly admitted clean rows.

    Inputs are canonical clean curves and once-transformed parameters. There
    is deliberately no augmented-array argument to this ASTRAI fit interface.
    Refit calls this function again with a refit Partition, never inner objects.
    """
    cfg = {} if cfg is None else cfg
    curves, parameters = as_model_array(curves), as_model_array(transformed_parameters)
    if curves.ndim != 2 or parameters.ndim != 2 or len(curves) != len(parameters) or len(curves) != partition.n_samples:
        raise ValueError("Canonical arrays and partition must have matching sample counts")
    validate_pca_size(n_components, len(partition.training), curves.shape[1])
    rows = partition.indices("training")
    clean, targets = curves[rows], parameters[rows]
    if not np.isfinite(clean).all() or not np.isfinite(targets).all():
        raise ValueError("Training values must be finite")
    if not np.any(np.var(clean, axis=0) > 0):
        raise ValueError("Clean training curves must have non-zero variance for PCA")
    x_scaler = StandardScaler().fit(clean)
    y_scaler = StandardScaler().fit(targets)
    pca = PCA(n_components=n_components, random_state=pca_seed).fit(x_scaler.transform(clean))
    manifest = {"bundle_schema_version": BUNDLE_SCHEMA_VERSION,
                "dataset_id": data_id or dataset_identity(curves, parameters),
                "partition": partition.record(), "fit_policy": FIT_POLICY,
                "n_features": curves.shape[1], "pca_components": int(n_components),
                "pca_seed": int(pca_seed), **parameter_semantics(cfg, parameters.shape[1])}
    objects = {"x_scaler": x_scaler, "y_scaler": y_scaler, "pca": pca}
    manifest["learned_state"] = {name: _learned_state_digest(obj) for name, obj in objects.items()}
    manifest["bundle_id"] = _digest(manifest)
    association = {key: manifest[key] for key in ("bundle_id", "dataset_id", "fit_policy", "target_transform", "param_names", "parameter_units", "unit_convention")}
    for obj in objects.values():
        obj.astrai_association = association.copy()
    return PreprocessingBundle(x_scaler, y_scaler, pca, manifest)


def load_preprocessing_bundle(directory, *, partition=None, data_id=None, cfg=None,
                              n_components=None):
    """Reject stale or misassociated artefacts before using a fitted bundle."""
    directory = Path(directory)
    path = directory / "bundle.yaml"
    if not path.is_file():
        raise ValueError("Training requires pool-specific preprocessing; regenerate artefacts")
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Unsupported preprocessing bundle schema")
    recorded = Partition.from_record(manifest["partition"])
    identity = dict(manifest)
    bundle_id = identity.pop("bundle_id", None)
    if bundle_id != _digest(identity) or manifest.get("fit_policy") != FIT_POLICY:
        raise ValueError("Invalid preprocessing bundle identity or fit policy")
    if partition is not None and recorded != partition:
        raise ValueError("Preprocessing bundle belongs to a different pool, fold or role")
    if data_id is not None and manifest["dataset_id"] != data_id:
        raise ValueError("Preprocessing bundle belongs to a different dataset")
    if n_components is not None and manifest["pca_components"] != n_components:
        raise ValueError("Preprocessing PCA configuration differs")
    if cfg is not None:
        for key, value in parameter_semantics(cfg, manifest["n_params"]).items():
            if manifest.get(key) != value:
                raise ValueError(f"Preprocessing parameter semantics differ: {key}")
    objects = {}
    for name in ("x_scaler", "y_scaler", "pca"):
        obj = joblib.load(directory / f"{name}.pkl")
        if getattr(obj, "astrai_association", {}).get("bundle_id") != bundle_id or _learned_state_digest(obj) != manifest["learned_state"][name]:
            raise ValueError(f"Preprocessing object does not match its bundle: {name}")
        objects[name] = obj
    return PreprocessingBundle(**objects, manifest=manifest)


def load_training_source(directory, cfg):
    """Load recoverable canonical arrays and validate the shared training plan."""
    from astrai.utils.partitions import (holdout_partition, outer_partitions,
                                        partition_seed, resolve_validation_fraction, selection_plan)
    directory = Path(directory)
    metadata_path = directory / "metadata.yaml"
    if not metadata_path.is_file():
        raise ValueError("Training requires pool-specific preprocessing; regenerate artefacts")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("preprocessing_artefact_schema_version") != PREPROCESSING_ARTEFACT_SCHEMA_VERSION or metadata.get("run", {}).get("status") != "completed":
        raise ValueError("Training requires completed schema 7 preprocessing; regenerate artefacts")
    for name in ("x_raw.npy", "y_physical.npy", "y_transformed.npy"):
        path = directory / name
        expected_digest = metadata.get("array_artefacts", {}).get(name, {}).get("sha256")
        if not path.is_file() or expected_digest != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Canonical source differs from its manifest: {name}")
    curves = as_model_array(np.load(directory / "x_raw.npy", allow_pickle=False))
    parameters = as_model_array(np.load(directory / "y_transformed.npy", allow_pickle=False))
    if curves.ndim != 2 or parameters.ndim != 2 or len(curves) != len(parameters):
        raise ValueError("Canonical source arrays must contain aligned rows")
    if cfg.get("data", {}).get("n_days", curves.shape[1]) != curves.shape[1]:
        raise ValueError("Configured curve length differs from preprocessing")
    expected_views = view_configuration(cfg)
    if metadata.get("view_configuration") != expected_views:
        raise ValueError("Augmentation configuration differs from precomputed training views")
    data_id = dataset_identity(curves, parameters)
    if metadata.get("dataset", {}).get("id") != data_id:
        raise ValueError("Canonical dataset differs from preprocessing metadata")
    plan = yaml.safe_load((directory / "partitions.yaml").read_text(encoding="utf-8"))
    fraction = resolve_validation_fraction(cfg)
    settings = cfg["preprocessing"]
    if (plan.get("schema_version") != 1 or plan.get("dataset_id") != data_id
            or plan.get("protocol") != "holdout_validation"
            or plan.get("validation_fraction") != fraction
            or plan.get("outer_folds") != settings["n_splits"]
            or plan.get("base_seed") != settings["random_seed"]
            or plan.get("selection_folds") != cfg.get("partitioning", {}).get("selection_folds")):
        raise ValueError("Configuration does not match the preprocessing partition plan")
    expected = [holdout_partition(len(curves), dev, test, fraction,
                                 partition_seed(settings["random_seed"], fold), fold)
                for fold, (dev, test) in enumerate(outer_partitions(
                    len(curves), settings["n_splits"], settings["random_seed"]), 1)]
    if [p.record() for p in expected] != plan.get("holdout"):
        raise ValueError("Preprocessing partitions differ from the configured plan")
    expected_selections = selection_plan(expected, plan["selection_folds"], settings["random_seed"])
    if {key: [part.record() for part in parts] for key, parts in expected_selections.items()} != plan.get("selection"):
        raise ValueError("Selection assignments differ from the configured plan")
    return {"curves": curves, "parameters": parameters, "dataset_id": data_id,
            "partitions": expected, "metadata": metadata}


def load_training_fold(directory, cfg, fold, source):
    """Load one verified bundle; reject swapped arrays and index mappings."""
    partition = source["partitions"][fold - 1]
    fold_dir = Path(directory) / f"fold_{fold}"
    for name, role in (("train_idx", "pool"), ("training_idx", "training"),
                       ("validation_idx", "validation"), ("test_idx", "test")):
        values = np.load(fold_dir / f"{name}.npy", allow_pickle=False)
        if not np.array_equal(values, partition.indices(role)):
            raise ValueError(f"Preprocessing index mapping differs: {name}")
    for path in fold_dir.glob("*.npy"):
        record = source["metadata"].get("array_artefacts", {}).get(f"fold_{fold}/{path.name}", {})
        if record.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Preprocessing array differs from its manifest: {path.name}")
    bundle = load_preprocessing_bundle(fold_dir, partition=partition,
                                      data_id=source["dataset_id"], cfg=cfg,
                                      n_components=cfg["preprocessing"]["pca_components"])
    if source["metadata"].get("bundles", {}).get(f"fold_{fold}", {}).get("bundle_id") != bundle.manifest["bundle_id"]:
        raise ValueError("Fold bundle differs from preprocessing run metadata")
    return partition, bundle
