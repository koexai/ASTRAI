"""Sample-level partitions shared by model stages and preprocessing roles.

Selection uses the same K_select in outer-development and final selection.
The executable training workflow currently uses explicit holdout validation;
these partition builders do not orchestrate budget selection or model refits.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from sklearn.model_selection import KFold

from astrai.utils.array_dtypes import as_index_array
from astrai.utils.reproducibility import build_partition_seed, validate_seed


PARTITION_SCHEMA_VERSION = 1


def partition_seed(base_seed, outer_fold=None):
    """Derive a stage-independent stream; zero denotes final selection."""
    return build_partition_seed(base_seed, outer_fold)


def resolve_validation_fraction(cfg):
    """Resolve shared holdout size, accepting only agreeing legacy settings."""
    partitioning = cfg.get("partitioning", {})
    values = []
    if "validation_fraction" in partitioning:
        values.append(partitioning["validation_fraction"])
    for training in [cfg.get("training", {})] + [
        cfg.get(stage, {}).get("training", {})
        for stage in ("characterizer", "generator")
    ]:
        if "validation_fraction" in training:
            values.append(training["validation_fraction"])
    if not values:
        values = [0.1]
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value < 1:
            raise ValueError("Shared validation_fraction must be between 0 and 1")
    if any(value != values[0] for value in values):
        raise ValueError("Shared and legacy validation_fraction settings must agree")
    return float(values[0])


def split_development_indices(n_samples, validation_fraction, seed,
                              minimum_validation_samples=1):
    """Split development rows, retaining the historical holdout algorithm."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
        raise TypeError("Development sample count must be an integer.")
    if n_samples < 2:
        raise ValueError("At least two development samples are required for validation.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    validation_size = max(int(minimum_validation_samples),
                          int(np.ceil(n_samples * validation_fraction)))
    if validation_size >= n_samples:
        raise ValueError("validation_fraction leaves no samples for fold training.")
    permutation = np.random.default_rng(seed).permutation(n_samples)
    return np.sort(permutation[validation_size:]), np.sort(permutation[:validation_size])


def _indices(values, n_samples):
    # Lists from manifests may legitimately describe an empty validation/test.
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)
    result = as_index_array(values)
    if result.ndim != 1 or len(np.unique(result)) != len(result):
        raise ValueError("Partition indices must be one-dimensional and unique")
    if np.any(result < 0) or np.any(result >= n_samples):
        raise ValueError("Partition indices are outside the dataset")
    return result


@dataclass(frozen=True)
class Partition:
    """An explicit allowed pool and its roles, in canonical dataset row order."""
    n_samples: int
    pool: tuple
    training: tuple
    validation: tuple = ()
    test: tuple = ()
    role: str = "holdout_validation"
    outer_fold: int | None = None
    selection_fold: int | None = None

    def __post_init__(self):
        if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, Integral) or self.n_samples < 1:
            raise ValueError("Dataset sample count must be a positive integer")
        arrays = {}
        for name in ("pool", "training", "validation", "test"):
            array = _indices(getattr(self, name), self.n_samples)
            arrays[name] = array
            object.__setattr__(self, name, tuple(int(i) for i in array))
        train, val, test = (arrays[name] for name in ("training", "validation", "test"))
        if len(train) == 0:
            raise ValueError("Preprocessing requires a non-empty training pool")
        if any(np.intersect1d(a, b).size for a, b in ((train, val), (train, test), (val, test))):
            raise ValueError("Training, validation and test must be disjoint")
        if set(self.pool) != set(self.training) | set(self.validation):
            raise ValueError("Training and validation must cover the allowed pool")
        if self.role not in {"holdout_validation", "inner_selection", "final_selection", "outer_refit", "final_refit"}:
            raise ValueError("Unknown preprocessing role")
        for fold in (self.outer_fold, self.selection_fold):
            if fold is not None and (isinstance(fold, bool) or not isinstance(fold, Integral) or fold < 1):
                raise ValueError("Fold identifiers must be positive integers")
        if self.role in {"inner_selection", "outer_refit", "holdout_validation"} and self.outer_fold is None:
            raise ValueError("This role requires an outer fold")
        if self.role in {"final_selection", "final_refit"} and (self.outer_fold is not None or self.test):
            raise ValueError("Final roles have no outer fold or internal test")
        if self.role in {"inner_selection", "final_selection"}:
            if self.selection_fold is None or not self.validation:
                raise ValueError("Selection requires a fold identifier and validation rows")
        elif self.selection_fold is not None:
            raise ValueError("Only K-fold selection has a selection fold identifier")
        if self.role.endswith("refit") and self.validation:
            raise ValueError("Refit uses the entire allowed pool without validation")
        if self.role == "holdout_validation" and not self.validation:
            raise ValueError("Holdout validation must not be empty")

    def indices(self, name):
        return np.asarray(getattr(self, name), dtype=np.int64)

    def record(self):
        return {"schema_version": PARTITION_SCHEMA_VERSION,
                "n_samples": int(self.n_samples), "role": self.role,
                "outer_fold": self.outer_fold, "selection_fold": self.selection_fold,
                **{name: list(getattr(self, name)) for name in ("pool", "training", "validation", "test")}}

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        if values.pop("schema_version", None) != PARTITION_SCHEMA_VERSION:
            raise ValueError("Unsupported partition schema")
        return cls(**values)


def outer_partitions(n_samples, n_splits, base_seed):
    """Preserve the original shuffled outer KFold and canonical row ordering."""
    return KFold(n_splits=n_splits, shuffle=True,
                 random_state=validate_seed(base_seed)).split(np.arange(n_samples))


def holdout_partition(n_samples, development, test, fraction, seed, outer_fold):
    """Build the shared transitional holdout (at least two validation rows)."""
    development = _indices(development, n_samples)
    train, val = split_development_indices(len(development), fraction, seed, 2)
    return Partition(n_samples, tuple(development), tuple(development[train]),
                     tuple(development[val]), tuple(test), "holdout_validation", outer_fold)


def selection_partitions(n_samples, pool, k_select, base_seed, *, outer_fold=None, test=()):
    """Build every K_select fold, on outer-development or the final pool.

    Callers supply the same k_select in both contexts. No model is trained and
    no budget is chosen here. All derived views inherit these original rows.
    """
    pool = _indices(pool, n_samples)
    role = "final_selection" if outer_fold is None else "inner_selection"
    seed = partition_seed(base_seed, outer_fold)
    return [Partition(n_samples, tuple(pool), tuple(pool[train]), tuple(pool[val]),
                      tuple(test), role, outer_fold, fold)
            for fold, (train, val) in enumerate(outer_partitions(len(pool), k_select, seed), 1)]


def refit_partition(n_samples, pool, *, outer_fold=None, test=()):
    """Declare the full allowed refit pool; no inner state is transferred."""
    return Partition(n_samples, tuple(pool), tuple(pool), test=tuple(test),
                     role="final_refit" if outer_fold is None else "outer_refit",
                     outer_fold=outer_fold)


def selection_plan(outer_holdouts, k_select, base_seed):
    """Use one K_select for all outer selections and the full-data selection."""
    if k_select is None:
        return {}
    n_samples = outer_holdouts[0].n_samples
    result = {f"outer_{part.outer_fold}": selection_partitions(
        n_samples, part.pool, k_select, base_seed,
        outer_fold=part.outer_fold, test=part.test) for part in outer_holdouts}
    result["final"] = selection_partitions(n_samples, np.arange(n_samples), k_select, base_seed)
    return result
