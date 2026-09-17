import unittest
from unittest.mock import patch

import numpy as np

from astrai.utils.augmentation import apply_lsst_pipeline


class ApplyLsstPipelineTests(unittest.TestCase):
    def test_returns_interpolated_curves_and_retained_mask(self):
        curves = np.array([[0.0, 1.0, 2.0, 3.0]])
        original = curves.copy()

        with patch("astrai.utils.augmentation.generate_masking", return_value={
            "retained_mask": np.array([True, False, True, False]),
        }):
            augmented, retained_mask = apply_lsst_pipeline(
                curves,
                n_days=4,
                noise_std=0.0,
                samples_per_day=1,
            )

        np.testing.assert_array_equal(curves, original)
        np.testing.assert_allclose(
            augmented,
            np.array([[0.0, 1.0, 2.0, 2.0]]),
        )
        np.testing.assert_array_equal(
            retained_mask,
            np.array([[True, False, True, False]]),
        )
        self.assertEqual(retained_mask.dtype, np.dtype(bool))

    def test_keeps_the_existing_additive_noise_path(self):
        curves = np.array([[1.0, 2.0, 3.0]])
        noisy_curves = curves + 0.5

        with (
            patch(
                "astrai.utils.augmentation.add_gaussian_noise",
                return_value=noisy_curves,
            ) as additive_noise,
            patch(
                "astrai.utils.augmentation.add_exp_gaussian_log_noise"
            ) as logarithmic_noise,
            patch("astrai.utils.augmentation.generate_masking", return_value={
                "retained_mask": np.ones(3, dtype=bool),
            }),
        ):
            augmented, retained_mask = apply_lsst_pipeline(
                curves,
                n_days=3,
                noise_std=0.25,
                samples_per_day=1,
            )

        additive_noise.assert_called_once()
        logarithmic_noise.assert_not_called()
        self.assertEqual(additive_noise.call_args.args[1], 0.25)
        np.testing.assert_allclose(augmented, noisy_curves)
        np.testing.assert_array_equal(
            retained_mask,
            np.ones_like(curves, dtype=bool),
        )

    def test_zero_observations_raise_without_redrawing(self):
        with patch("astrai.utils.augmentation.generate_masking", return_value={
            "retained_mask": np.zeros(3, dtype=bool),
        }) as generate:
            with self.assertRaisesRegex(ValueError, "curve 0"):
                apply_lsst_pipeline(np.ones((1, 3)), 3, 0)
        generate.assert_called_once()

    def test_single_observation_is_constant_without_hidden_values(self):
        with patch("astrai.utils.augmentation.generate_masking", return_value={
            "retained_mask": np.array([False, True, False]),
        }):
            actual, _ = apply_lsst_pipeline(np.array([[100., 42., -100.]]), 3, 0)
        np.testing.assert_array_equal(actual, [[42., 42., 42.]])

    def test_real_pipeline_masks_agree_across_resolutions(self):
        a, ma = apply_lsst_pipeline(np.full((3, 421), 42.), 421, 0, 1, np.random.default_rng(42))
        b, mb = apply_lsst_pipeline(np.full((3, 1681), 42.), 1681, 0, 4, np.random.default_rng(42))
        np.testing.assert_array_equal(ma, mb[:, ::4])
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())

    def test_invalid_shape_and_values_fail_before_augmentation(self):
        for values, n in [(np.ones((2, 3)), 4), (np.ones(3), 3), (np.array([[np.nan]]), 1)]:
            with self.assertRaises(ValueError):
                apply_lsst_pipeline(values, n, 0)


if __name__ == "__main__":
    unittest.main()
