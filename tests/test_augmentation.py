import unittest
from unittest.mock import patch

import numpy as np

from utils.augmentation import apply_lsst_pipeline


class ApplyLsstPipelineTests(unittest.TestCase):
    def test_returns_interpolated_curves_and_retained_mask(self):
        curves = np.array([[0.0, 1.0, 2.0, 3.0]])
        original = curves.copy()

        with (
            patch(
                "utils.augmentation.lsst.sun_masking_np",
                return_value=np.zeros(4),
            ),
            patch(
                "utils.augmentation.lsst.random_cloud_masking",
                return_value=np.array([0.0, 1.0, 0.0, 1.0]),
            ),
        ):
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
                "utils.augmentation.add_gaussian_noise",
                return_value=noisy_curves,
            ) as additive_noise,
            patch(
                "utils.augmentation.add_exp_gaussian_log_noise"
            ) as logarithmic_noise,
            patch(
                "utils.augmentation.lsst.sun_masking_np",
                return_value=np.zeros(3),
            ),
            patch(
                "utils.augmentation.lsst.random_cloud_masking",
                return_value=np.zeros(3),
            ),
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


if __name__ == "__main__":
    unittest.main()
