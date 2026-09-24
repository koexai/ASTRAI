"""Configured recipes, temporary zero exclusion and independent masking."""
import unittest
from unittest.mock import patch

import numpy as np

from astrai.utils import augmentation as aug
from astrai.utils.reproducibility import (
    build_augmentation_seed_plan, make_augmentation_rngs,
    build_training_seed_plan, build_unified_preprocessing_seed_plan,
)


IID = {"model": "iid_log10", "sigma_dex": .05}
HETERO = {"model": "heteroscedastic_normalised", "a": .01, "b": .0004}


class ConfiguredAugmentationTests(unittest.TestCase):
    def test_zero_exclusion_preserves_nonzero_draws_and_rng_state(self):
        values = np.array([[0., 38., 42.], [41., 0., 1e-12]])
        before = values.copy()
        actual_rng, expected_rng = np.random.default_rng(42), np.random.default_rng(42)
        actual = aug.apply_noise(values, HETERO, rng=actual_rng)
        expected = aug.add_heteroscedastic_noise_in_normalised_luminosity(
            values[values != 0], a=.01, b=.0004, rng=expected_rng)
        np.testing.assert_array_equal(actual[values != 0], expected)
        np.testing.assert_array_equal(actual[values == 0], 0)
        np.testing.assert_array_equal(values, before)
        self.assertEqual(actual_rng.bit_generator.state, expected_rng.bit_generator.state)
        self.assertEqual(actual.dtype, np.float64)
        self.assertEqual(aug.augmentation_record(values, HETERO, 42)["excluded_zero_count"], 2)

    def test_all_zero_consumes_no_noise_rng_but_still_validates(self):
        rng = np.random.default_rng(42)
        before = rng.bit_generator.state
        np.testing.assert_array_equal(aug.apply_noise(np.zeros((2, 5)), HETERO, rng=rng), 0)
        self.assertEqual(rng.bit_generator.state, before)
        with self.assertRaises(ValueError):
            aug.apply_noise(np.zeros((2, 5)), {**HETERO, "a": -1}, rng=rng)
        direct = aug.add_heteroscedastic_noise_in_normalised_luminosity(
            np.zeros(10), a=.01, b=.0004, rng=rng)
        self.assertTrue(np.all(direct > 0))

    def test_zero_preservation_ends_before_interpolation(self):
        with patch.object(aug, "generate_masking", return_value={
                "retained_mask": np.array([True, False, True])}):
            values, mask = aug.apply_augmentation([[41., 0., 43.]], 3,
                {**HETERO, "a": 0, "b": 0}, seed=42)
        np.testing.assert_array_equal(values, [[41., 42., 43.]])
        self.assertFalse(mask[0, 1])

    def test_modern_masks_are_independent_of_noise_and_zero_count(self):
        values = np.full((3, 421), 38.)
        configurations = [IID, {**IID, "sigma_dex": 0}, HETERO, {**HETERO, "b": .1}]
        masks = []
        for config in configurations:
            first = aug.apply_augmentation(values, 421, config, seed=42)
            again = aug.apply_augmentation(values, 421, config, seed=42)
            np.testing.assert_array_equal(first[0], again[0])
            np.testing.assert_array_equal(first[1], again[1])
            masks.append(first[1])
        zero_values = values.copy()
        zero_values[:, ::2] = 0
        masks.append(aug.apply_augmentation(zero_values, 421, HETERO, seed=42)[1])
        for mask in masks[1:]:
            np.testing.assert_array_equal(mask, masks[0])

    def test_modern_masks_agree_across_grids_with_resampling(self):
        coarse = aug.apply_augmentation(np.full((2, 401), 38.), 401, HETERO,
                                        seed=42, samples_per_day=1)
        fine = aug.apply_augmentation(np.full((2, 1601), 38.), 1601, HETERO,
                                      seed=42, samples_per_day=4)
        np.testing.assert_array_equal(coarse[1], fine[1][:, ::4])

    def test_legacy_kernels_and_shared_stream_are_preserved(self):
        values = np.full((3, 421), 42.)
        cases = [("legacy_iid_gaussian", {"noise_std": .05}, aug.add_gaussian_noise_slow),
                 ("legacy_tiled_gaussian", {"noise_std": 0}, aug.add_gaussian_noise),
                 ("legacy_exp_sqrt_gaussian", {"sigma": .05}, aug.add_exp_gaussian_log_noise)]
        for name, parameters, kernel in cases:
            with self.subTest(model=name):
                expected_rng = np.random.default_rng(42)
                expected = kernel(values.copy(), rng=expected_rng, **parameters)
                if isinstance(expected, tuple):
                    expected = expected[0]
                masks = []
                for row in range(len(values)):
                    mask = aug.generate_masking(aug.build_time_axis(421), rng=expected_rng)["retained_mask"]
                    masks.append(mask)
                    expected[row] = aug.interpolate_observations(expected[row], mask, aug.build_time_axis(421))
                actual = aug.apply_augmentation(values, 421, {"model": name, **parameters}, seed=42)
                np.testing.assert_array_equal(actual[0], expected)
                np.testing.assert_array_equal(actual[1], masks)

    def test_versioned_plan_reconstructs_streams_and_rejects_tampering(self):
        for modern in [False, True]:
            plan = build_augmentation_seed_plan(42, modern=modern)
            self.assertEqual(plan["policy_version"], 1)
            self.assertEqual(plan["derivation_scheme_version"], 1)
            left, right = make_augmentation_rngs(plan)
            self.assertEqual(left is right, not modern)
            with self.assertRaises(ValueError):
                make_augmentation_rngs({**plan, "policy_version": 99})
        self.assertEqual(build_unified_preprocessing_seed_plan(42, 1)["augmentation"], 1385871029)
        before = build_training_seed_plan(42, "characterizer", 1)
        build_augmentation_seed_plan(42, modern=True)
        self.assertEqual(before, build_training_seed_plan(42, "characterizer", 1))

    def test_global_state_and_input_unchanged_and_empty_batch_validated(self):
        values = np.full((2, 421), 42.)
        np.random.seed(123)
        expected = np.random.random()
        np.random.seed(123)
        aug.apply_augmentation(values, 421, IID, seed=42)
        self.assertEqual(np.random.random(), expected)
        np.testing.assert_array_equal(values, 42.)
        curves, mask = aug.apply_augmentation(np.empty((0, 421)), 421, IID, seed=42)
        self.assertEqual(curves.shape, mask.shape)
        with self.assertRaises(ValueError):
            aug.apply_augmentation(np.empty((0, 421)), 421, {}, seed=42)


if __name__ == "__main__":
    unittest.main()
