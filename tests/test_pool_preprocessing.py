"""Numerical exclusion, clean-only policy and role/partition contracts."""
import copy
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import KFold
import yaml

from astrai.utils.partitions import (
    Partition, holdout_partition, outer_partitions, partition_seed,
    refit_partition, resolve_validation_fraction, selection_partitions, selection_plan,
)
from astrai.utils.preprocessing import (
    dataset_identity, fit_preprocessing, load_preprocessing_bundle, load_training_source,
    validate_pca_size,
)
from astrai.utils.reproducibility import build_role_preprocessing_seed_plan
from astrai.utils.target_transformations import validate_parameter_scalers


class PoolPreprocessingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(91)
        self.curves = rng.normal(size=(18, 6)).astype(np.float32)
        self.parameters = np.log1p(rng.uniform(0, 10, (18, 2))).astype(np.float32)
        self.cfg = {"data": {"n_params": 2, "param_names": ["Mass", "Energy"]}}
        self.partition = Partition(18, tuple(range(12)), tuple(range(8)),
                                   tuple(range(8, 12)), tuple(range(12, 18)), outer_fold=1)

    def fit(self, partition=None, curves=None, parameters=None):
        return fit_preprocessing(self.curves if curves is None else curves,
                                 self.parameters if parameters is None else parameters,
                                 self.partition if partition is None else partition,
                                 3, 42, self.cfg)

    def assert_same_state(self, left, right):
        np.testing.assert_array_equal(left.x_scaler.mean_, right.x_scaler.mean_)
        np.testing.assert_array_equal(left.x_scaler.scale_, right.x_scaler.scale_)
        np.testing.assert_array_equal(left.y_scaler.mean_, right.y_scaler.mean_)
        np.testing.assert_array_equal(left.y_scaler.scale_, right.y_scaler.scale_)
        np.testing.assert_array_equal(left.pca.components_, right.pca.components_)
        np.testing.assert_array_equal(left.pca.mean_, right.pca.mean_)

    def test_excluded_validation_and_test_do_not_influence_fit(self):
        first = self.fit()
        for indices in (self.partition.validation, self.partition.test):
            curves, parameters = self.curves.copy(), self.parameters.copy()
            curves[list(indices)] += 1000
            parameters[list(indices)] += 100
            other = self.fit(curves=curves, parameters=parameters)
            self.assert_same_state(first, other)
            np.testing.assert_array_equal(first.transform_curves(self.curves[:8]),
                                          other.transform_curves(self.curves[:8]))

    def test_training_perturbation_changes_learned_state(self):
        first = self.fit()
        curves, parameters = self.curves.copy(), self.parameters.copy()
        curves[0] += 100
        parameters[0] += 10
        other = self.fit(curves=curves, parameters=parameters)
        self.assertFalse(np.array_equal(first.x_scaler.mean_, other.x_scaler.mean_))
        self.assertFalse(np.array_equal(first.y_scaler.mean_, other.y_scaler.mean_))
        self.assertFalse(np.allclose(first.pca.components_, other.pca.components_))

    def test_clean_only_policy_is_independent_of_augmented_views(self):
        first = self.fit()
        state = copy.deepcopy(first)
        for seed in (42, 93):
            augmented = self.curves[:8] + np.random.default_rng(seed).normal(size=(8, 6)) * 100
            output = first.transform_curves(augmented)
            self.assertEqual(output.shape, (8, 3))
            self.assert_same_state(first, state)
        self.assertEqual(first.manifest["fit_policy"], "clean_original_samples")
        self.assertEqual(int(first.x_scaler.n_samples_seen_), 8)
        self.assertEqual(int(first.y_scaler.n_samples_seen_), 8)

    def test_final_refit_uses_fresh_objects_and_all_rows(self):
        inner = self.fit()
        final = self.fit(refit_partition(18, range(18)))
        for name in ("x_scaler", "y_scaler", "pca"):
            self.assertIsNot(getattr(inner, name), getattr(final, name))
        np.testing.assert_allclose(final.x_scaler.mean_, self.curves.mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(final.y_scaler.mean_, self.parameters.mean(axis=0), atol=1e-6)
        self.assertEqual(int(final.x_scaler.n_samples_seen_), 18)
        self.assertEqual(final.manifest["partition"]["role"], "final_refit")

    def test_outer_refit_excludes_outer_test(self):
        partition = refit_partition(18, range(12), outer_fold=1, test=range(12, 18))
        first = self.fit(partition)
        changed = self.curves.copy()
        changed[12:] += 1000
        self.assert_same_state(first, self.fit(partition, curves=changed))
        self.assertEqual(int(first.x_scaler.n_samples_seen_), 12)

    def test_all_selection_folds_exclude_their_validation(self):
        for outer in (None, 1):
            pool, test = (range(18), ()) if outer is None else (range(12), range(12, 18))
            for part in selection_partitions(18, pool, 3, 42, outer_fold=outer, test=test):
                first = self.fit(part)
                curves, parameters = self.curves.copy(), self.parameters.copy()
                excluded = list(part.validation + part.test)
                curves[excluded] += 100
                parameters[excluded] += 50
                self.assert_same_state(first, self.fit(part, curves, parameters))

    def test_serialised_bundle_preserves_transformations(self):
        first = self.fit()
        with tempfile.TemporaryDirectory() as path:
            first.save(path)
            loaded = load_preprocessing_bundle(path, partition=self.partition,
                                               data_id=dataset_identity(self.curves, self.parameters),
                                               cfg=self.cfg, n_components=3)
            self.assert_same_state(first, loaded)
            np.testing.assert_array_equal(first.transform_curves(self.curves), loaded.transform_curves(self.curves))
            self.assertEqual(loaded.transform_parameters(self.parameters).dtype, np.float32)

    def test_bundle_rejects_wrong_pool_role_and_fold(self):
        with tempfile.TemporaryDirectory() as path:
            self.fit().save(path)
            alternatives = [refit_partition(18, range(12), outer_fold=1, test=range(12, 18)),
                            Partition(18, tuple(range(12)), tuple(range(8)), tuple(range(8, 12)),
                                      tuple(range(12, 18)), outer_fold=2),
                            Partition(18, tuple(range(12)), tuple(range(1, 9)), (0, 9, 10, 11),
                                      tuple(range(12, 18)), outer_fold=1)]
            for part in alternatives:
                with self.assertRaisesRegex(ValueError, "pool, fold or role"):
                    load_preprocessing_bundle(path, partition=part)

    def test_bundle_rejects_dataset_order_and_parameter_order_mismatches(self):
        with tempfile.TemporaryDirectory() as path:
            self.fit().save(path)
            with self.assertRaisesRegex(ValueError, "dataset"):
                load_preprocessing_bundle(path, data_id=dataset_identity(self.curves[::-1], self.parameters[::-1]))
            cfg = copy.deepcopy(self.cfg)
            cfg["data"]["param_names"].reverse()
            with self.assertRaisesRegex(ValueError, "semantics"):
                load_preprocessing_bundle(path, cfg=cfg)

    def test_bundle_rejects_swapped_learned_objects(self):
        with tempfile.TemporaryDirectory() as path:
            self.fit().save(path)
            other = self.fit(refit_partition(18, range(18)))
            joblib.dump(other.pca, Path(path) / "pca.pkl")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_preprocessing_bundle(path)

    def test_bundle_rejects_mutated_state_with_original_association(self):
        with tempfile.TemporaryDirectory() as path:
            bundle = self.fit()
            bundle.save(path)
            bundle.y_scaler.mean_[0] += 1
            joblib.dump(bundle.y_scaler, Path(path) / "y_scaler.pkl")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_preprocessing_bundle(path)

    def test_bundle_rejects_modified_manifest(self):
        with tempfile.TemporaryDirectory() as path:
            self.fit().save(path)
            file = Path(path) / "bundle.yaml"
            manifest = yaml.safe_load(file.read_text())
            manifest["pca_seed"] += 1
            file.write_text(yaml.safe_dump(manifest))
            with self.assertRaisesRegex(ValueError, "identity"):
                load_preprocessing_bundle(path)

    def test_bundle_cannot_overwrite_existing_artefacts(self):
        with tempfile.TemporaryDirectory() as path:
            bundle = self.fit()
            bundle.save(path)
            with self.assertRaises(FileExistsError):
                bundle.save(path)

    def test_parameter_handoff_checks_scaling_and_semantics(self):
        first, second = self.fit(), self.fit()
        validate_parameter_scalers(first.y_scaler, second.y_scaler)
        different = self.fit(refit_partition(18, range(18)))
        with self.assertRaises(ValueError):
            validate_parameter_scalers(first.y_scaler, different.y_scaler)
        second.y_scaler.astrai_association["parameter_units"] = ["other", "other"]
        with self.assertRaisesRegex(ValueError, "parameter_units"):
            validate_parameter_scalers(first.y_scaler, second.y_scaler)

    def test_pca_preflight_rejects_small_or_degenerate_training(self):
        for n_pca, size in ((4, 3), (0, 8), (3, 1), (True, 8)):
            with self.assertRaises(ValueError):
                validate_pca_size(n_pca, size, 6)
        with self.assertRaisesRegex(ValueError, "variance"):
            self.fit(curves=np.ones((18, 6)))

    def test_old_or_incomplete_preprocessing_is_rejected_for_training(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError, "regenerate"):
                load_training_source(path, {})
            for version, status in ((5, "completed"), (6, "failed"), (6, "running")):
                (Path(path) / "metadata.yaml").write_text(yaml.safe_dump({
                    "preprocessing_artefact_schema_version": version, "run": {"status": status}}))
                with self.assertRaisesRegex(ValueError, "regenerate"):
                    load_training_source(path, {})


class SharedPartitionTests(unittest.TestCase):
    def test_outer_partitions_preserve_historical_assignments(self):
        for actual, expected in zip(outer_partitions(23, 4, 42),
                                    KFold(4, shuffle=True, random_state=42).split(range(23))):
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left, right)

    def test_holdout_is_shared_and_independent_of_execution_order(self):
        parts = {}
        for stage in ("generator", "characterizer"):
            parts[stage] = holdout_partition(18, range(12), range(12, 18), 0.2, partition_seed(42, 1), 1)
        self.assertEqual(parts["generator"], parts["characterizer"])
        part = parts["generator"]
        self.assertEqual(set(part.training) | set(part.validation), set(range(12)))
        self.assertFalse(set(part.training) & set(part.validation))

    def test_final_selection_is_kfold_over_the_whole_pool(self):
        folds = selection_partitions(18, range(18), 3, 42)
        self.assertEqual(len(folds), 3)
        self.assertEqual(sorted(i for p in folds for i in p.validation), list(range(18)))
        for index, part in enumerate(folds, 1):
            self.assertIsNone(part.outer_fold)
            self.assertEqual(part.selection_fold, index)
            self.assertEqual(part.role, "final_selection")
            self.assertEqual(len(part.training), 12)
            self.assertFalse(part.test)

    def test_one_k_select_applies_to_outer_and_final_selection(self):
        holdouts = [holdout_partition(18, dev, test, 0.2, partition_seed(42, fold), fold)
                    for fold, (dev, test) in enumerate(outer_partitions(18, 3, 42), 1)]
        plan = selection_plan(holdouts, 4, 42)
        self.assertEqual({len(folds) for folds in plan.values()}, {4})
        for part in plan["outer_1"]:
            self.assertFalse(set(part.training + part.validation) & set(part.test))
        self.assertEqual(plan, selection_plan(holdouts, 4, 42))

    def test_partition_rejects_overlap_duplicates_and_out_of_bounds(self):
        for train, val in (((0, 1), (1, 2)), ((0, 0), (1, 2)), ((0, 4), (1, 2))):
            with self.assertRaises(ValueError):
                Partition(4, (0, 1, 2), train, val, (3,), outer_fold=1)

    def test_refit_and_final_roles_reject_invalid_context(self):
        with self.assertRaises(ValueError):
            Partition(4, (0, 1, 2), (0, 1), (2,), (3,), "outer_refit", 1)
        with self.assertRaises(ValueError):
            refit_partition(4, range(3), test=(3,))
        with self.assertRaises(ValueError):
            Partition(4, (0, 1, 2), (0, 1), (2,), (3,), "inner_selection", 1)

    def test_partition_record_roundtrip(self):
        part = holdout_partition(18, range(12), range(12, 18), 0.2, 42, 1)
        self.assertEqual(part, Partition.from_record(part.record()))
        self.assertEqual(part.indices("training").dtype, np.int64)

    def test_legacy_fraction_must_agree_with_shared_configuration(self):
        cfg = {"partitioning": {"validation_fraction": 0.2},
               "characterizer": {"training": {"validation_fraction": 0.2}},
               "generator": {"training": {"validation_fraction": 0.2}}}
        self.assertEqual(resolve_validation_fraction(cfg), 0.2)
        cfg["generator"]["training"]["validation_fraction"] = 0.3
        with self.assertRaisesRegex(ValueError, "agree"):
            resolve_validation_fraction(cfg)

    def test_selection_and_refit_seeds_are_distinct_and_repeatable(self):
        plans = [build_role_preprocessing_seed_plan(42, role, outer, inner)
                 for role, outer, inner in (("inner_selection", 1, 1), ("inner_selection", 1, 2),
                                           ("outer_refit", 1, None), ("final_selection", None, 1),
                                           ("final_refit", None, None))]
        self.assertEqual(len({plan["pca"] for plan in plans}), 5)
        self.assertEqual(plans[0], build_role_preprocessing_seed_plan(42, "inner_selection", 1, 1))


if __name__ == "__main__":
    unittest.main()
