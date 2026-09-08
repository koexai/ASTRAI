import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from astrai.utils.target_transformations import (
    experiment_target_contract,
    load_fold_target_array,
    physical_to_scaled,
    physical_to_transformed,
    scaled_to_physical,
    target_transform_contract,
    transformed_to_physical,
    validate_parameter_scalers,
    validate_physical_targets,
    validate_preprocessing_target_contract,
    validate_target_contract_compatibility,
)


class DummyScaler:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)

    def transform(self, values):
        return (np.asarray(values) - self.mean_) / self.scale_

    def inverse_transform(self, values):
        return np.asarray(values) * self.scale_ + self.mean_


class TargetTransformationTests(unittest.TestCase):
    def test_contract_defaults_legacy_configs_to_log1p(self):
        self.assertEqual(target_transform_contract({})["name"], "log1p")

    def test_contract_accepts_explicit_log1p(self):
        contract = target_transform_contract(
            {"data": {"target_transform": "log1p"}}
        )
        self.assertEqual(contract["version"], 1)
        self.assertEqual(contract["physical_domain"], "non_negative")

    def test_contract_rejects_unavailable_transforms(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            target_transform_contract(
                {"data": {"target_transform": "identity"}}
            )

    def test_round_trip_preserves_zero_and_physical_ranges(self):
        for dtype in (np.float32, np.float64):
            for physical in (
                [0.0, 0.01, 0.5, 30.0],
                [0.0, 1e-5, 0.2, 1.0, 5.0, 25.0, 100.0],
            ):
                values = np.asarray([physical], dtype=dtype)
                recovered = transformed_to_physical(
                    physical_to_transformed(values)
                )
                np.testing.assert_allclose(recovered, values, rtol=1e-6)
                self.assertEqual(recovered.shape, values.shape)

    def test_inverse_does_not_clip_model_extrapolation(self):
        physical = transformed_to_physical(np.array([[-0.5]]))
        self.assertLess(physical[0, 0], 0.0)

    def test_rejects_negative_observed_physical_targets(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            physical_to_transformed(np.array([[1.0, -0.01]]))

    def test_rejects_non_finite_physical_targets(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_physical_targets(np.array([[value]]))

    def test_scaled_round_trip_returns_physical_values(self):
        scaler = DummyScaler(mean=[0.5, 1.0], scale=[2.0, 4.0])
        physical = np.array([[0.0, 10.0], [2.0, 5.0]])
        scaled = physical_to_scaled(physical, scaler)
        np.testing.assert_allclose(
            scaled_to_physical(scaled, scaler),
            physical,
        )

    def test_parameter_scaler_compatibility_checks_values(self):
        validate_parameter_scalers(
            DummyScaler([1.0, 2.0], [3.0, 4.0]),
            DummyScaler([1.0, 2.0], [3.0, 4.0]),
        )
        with self.assertRaisesRegex(ValueError, "mean_ values differ"):
            validate_parameter_scalers(
                DummyScaler([1.0, 2.0], [3.0, 4.0]),
                DummyScaler([1.0, 2.5], [3.0, 4.0]),
            )

    def test_contract_compatibility_checks_version(self):
        current = target_transform_contract()
        incompatible = dict(current, version=2)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            validate_target_contract_compatibility(current, incompatible)


class TargetMetadataCompatibilityTests(unittest.TestCase):
    def test_accepts_metadata_free_legacy_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            contract = validate_preprocessing_target_contract(temp_dir, {})
        self.assertEqual(contract["name"], "log1p")

    def test_rejects_ambiguous_versioned_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metadata.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "preprocessing_artefact_schema_version": 4,
                        "run": {"status": "completed"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Regenerate"):
                validate_preprocessing_target_contract(temp_dir, {})

    def test_accepts_schema_five_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metadata.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "preprocessing_artefact_schema_version": 5,
                        "run": {"status": "completed"},
                        "target_transform": target_transform_contract(),
                    }
                ),
                encoding="utf-8",
            )
            contract = validate_preprocessing_target_contract(temp_dir, {})
        self.assertEqual(contract["name"], "log1p")

    def test_experiment_metadata_v3_requires_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "metadata.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_metadata_version": 3,
                        "data": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not record"):
                experiment_target_contract(temp_dir, {})

    def test_experiment_metadata_v2_uses_legacy_log1p_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "metadata.yaml").write_text(
                yaml.safe_dump({"experiment_metadata_version": 2}),
                encoding="utf-8",
            )
            contract = experiment_target_contract(temp_dir, {})
        self.assertEqual(contract["name"], "log1p")

    def test_experiment_metadata_checks_parameter_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "metadata.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_metadata_version": 3,
                        "data": {
                            "n_params": 2,
                            "param_names": ["Energy", "Mass"],
                            "target_transform": target_transform_contract(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "param_names"):
                experiment_target_contract(
                    temp_dir,
                    {
                        "data": {
                            "n_params": 2,
                            "param_names": ["Mass", "Energy"],
                        }
                    },
                )

    def test_fold_loader_prefers_canonical_name_and_supports_legacy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fold_dir = Path(temp_dir)
            np.save(fold_dir / "y_test.npy", np.array([[1.0]]))
            np.testing.assert_array_equal(
                load_fold_target_array(
                    fold_dir,
                    "y_test_transformed.npy",
                    "y_test.npy",
                ),
                [[1.0]],
            )
            np.save(
                fold_dir / "y_test_transformed.npy",
                np.array([[2.0]]),
            )
            np.testing.assert_array_equal(
                load_fold_target_array(
                    fold_dir,
                    "y_test_transformed.npy",
                    "y_test.npy",
                ),
                [[2.0]],
            )


if __name__ == "__main__":
    unittest.main()
