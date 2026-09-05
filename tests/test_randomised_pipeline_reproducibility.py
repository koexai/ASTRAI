import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from joblib import load

from scripts import preprocess
from utils import lsst
from utils.augmentation import (
    add_gaussian_noise,
    add_exp_gaussian_log_noise,
    apply_lsst_pipeline,
)


class AugmentationReproducibilityTests(unittest.TestCase):
    def test_fast_noise_keeps_the_tiled_per_curve_pattern(self):
        curves = np.ones((3, 4))
        expected_rng = np.random.default_rng(42)
        expected_noise = np.tile(expected_rng.standard_normal(3), 4).reshape(
            curves.shape
        )

        actual = add_gaussian_noise(
            curves,
            noise_std=0.5,
            rng=np.random.default_rng(42),
        )

        np.testing.assert_array_equal(actual, curves + 0.5 * expected_noise)

    def test_explicit_rng_repeats_noise_and_masks(self):
        curves = np.arange(120, dtype=np.float64).reshape(4, 30) / 10

        first = apply_lsst_pipeline(
            curves,
            n_days=30,
            noise_std=0.05,
            samples_per_day=1,
            rng=np.random.default_rng(123),
        )
        repeated = apply_lsst_pipeline(
            curves,
            n_days=30,
            noise_std=0.05,
            samples_per_day=1,
            rng=np.random.default_rng(123),
        )

        np.testing.assert_array_equal(first[0], repeated[0])
        np.testing.assert_array_equal(first[1], repeated[1])

    def test_explicit_rng_does_not_mutate_numpy_global_state(self):
        curves = np.ones((2, 20))
        np.random.seed(2026)
        expected = np.random.random()
        np.random.seed(2026)

        apply_lsst_pipeline(
            curves,
            n_days=20,
            noise_std=0.05,
            samples_per_day=1,
            rng=np.random.default_rng(42),
        )

        self.assertEqual(np.random.random(), expected)

    def test_legacy_noise_seed_is_local_and_repeatable(self):
        values = np.log(np.array([[2.0, 3.0, 4.0]]))
        np.random.seed(2026)
        expected = np.random.random()
        np.random.seed(2026)

        first = add_exp_gaussian_log_noise(values, random_state=42)
        repeated = add_exp_gaussian_log_noise(values, random_state=42)

        np.testing.assert_array_equal(first[0], repeated[0])
        np.testing.assert_array_equal(first[1], repeated[1])
        self.assertEqual(np.random.random(), expected)

    def test_cloud_mask_rejects_seed_and_rng_together(self):
        with self.assertRaisesRegex(ValueError, "cannot be supplied together"):
            lsst.random_cloud_masking(
                np.ones(20),
                seed=42,
                rng=np.random.default_rng(42),
            )


class PreprocessingReproducibilityTests(unittest.TestCase):
    @staticmethod
    def _write_code_archive(run_dir, folder):
        del folder
        (Path(run_dir) / "code.zip").write_bytes(b"code")

    def test_two_runs_produce_identical_numpy_artefacts(self):
        cfg = {
            "data": {
                "n_days": 12,
                "samples_per_day": 1,
            },
            "preprocessing": {
                "pca_components": 3,
                "n_splits": 2,
                "random_seed": 42,
            },
            "augmentation": {"noise_std": 0.05},
        }
        x_raw = np.linspace(0.1, 4.8, 96).reshape(8, 12)
        y_raw = np.linspace(0.1, 1.6, 16).reshape(8, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_dir = root / "first"
            second_dir = root / "second"
            with (
                patch.object(preprocess, "_REPOSITORY_ROOT", root),
                patch.object(
                    preprocess,
                    "save_code",
                    side_effect=self._write_code_archive,
                ),
                patch.object(
                    preprocess,
                    "load_data",
                    return_value=(x_raw, y_raw),
                ),
                redirect_stdout(StringIO()),
            ):
                preprocess.run_preprocessing(cfg, out_dir=first_dir)
                preprocess.run_preprocessing(cfg, out_dir=second_dir)

            first_paths = sorted(
                path.relative_to(first_dir) for path in first_dir.rglob("*.npy")
            )
            second_paths = sorted(
                path.relative_to(second_dir)
                for path in second_dir.rglob("*.npy")
            )
            self.assertEqual(first_paths, second_paths)

            for relative_path in first_paths:
                with self.subTest(path=relative_path):
                    np.testing.assert_array_equal(
                        np.load(first_dir / relative_path),
                        np.load(second_dir / relative_path),
                    )

            first_metadata = yaml.safe_load(
                (first_dir / "metadata.yaml").read_text(encoding="utf-8")
            )
            second_metadata = yaml.safe_load(
                (second_dir / "metadata.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                first_metadata["preprocessing"]["seed_plan"],
                second_metadata["preprocessing"]["seed_plan"],
            )
            self.assertEqual(
                first_metadata["preprocessing_artefact_schema_version"],
                3,
            )
            self.assertEqual(
                load(first_dir / "pca.pkl").random_state,
                first_metadata["preprocessing"]["seed_plan"]["pca"],
            )


if __name__ == "__main__":
    unittest.main()
