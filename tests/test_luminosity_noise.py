"""Scientific and numerical contracts for direct bolometric noise kernels."""
import copy
import math
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from astrai.utils import augmentation as aug


class ScriptedGenerator(np.random.Generator):
    """Exercise rejection and arithmetic failures without chance events."""

    def __init__(self, draws):
        super().__init__(np.random.PCG64(42))
        self.draws = iter(draws)
        self.sizes = []

    def standard_normal(self, size=None):
        self.sizes.append(size)
        return np.broadcast_to(next(self.draws), size).copy()


class LuminosityNoiseTests(unittest.TestCase):
    @staticmethod
    def calls(values, rng=None):
        return (
            lambda: aug.add_iid_gaussian_noise_in_log10_luminosity(
                values, sigma_dex=0.05, rng=rng),
            lambda: aug.add_heteroscedastic_noise_in_normalised_luminosity(
                values, a=0.01, b=0.0004, rng=rng),
        )

    def test_shape_dtype_and_no_input_mutation(self):
        for shape in ((12,), (3, 4)):
            original = np.linspace(39, 44, 12, dtype=np.float32).reshape(shape)
            for call in self.calls(original, np.random.default_rng(5)):
                before = original.copy()
                result = call()
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, np.dtype("float64"))
                self.assertTrue(np.isfinite(result).all())
                self.assertFalse(np.shares_memory(result, original))
                np.testing.assert_array_equal(original, before)

    def test_lists_and_empty_arrays(self):
        for values in ([40, 41, 42], np.empty((0,)), np.empty((0, 4)), np.empty((2, 0))):
            for call in self.calls(values):
                result = call()
                self.assertEqual(result.shape, np.shape(values))
                self.assertEqual(result.dtype, np.dtype("float64"))

    def test_rejects_invalid_array_contract(self):
        for values in (42., [[[42.]]], [np.nan], [np.inf], [-np.inf],
                       [42+1j], [True], ["42"], np.array([42], dtype=object)):
            for call in self.calls(values):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    call()

    def test_rejects_invalid_amplitudes(self):
        for bad in (-1, np.nan, np.inf, True, "0.1", [0.1], 1j):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                aug.add_iid_gaussian_noise_in_log10_luminosity([42], sigma_dex=bad)
            for name in ("a", "b"):
                kwargs = {"a": .01, "b": .0004, name: bad}
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    aug.add_heteroscedastic_noise_in_normalised_luminosity([42], **kwargs)

    def test_rejects_invalid_reference_and_rng(self):
        for bad in (np.nan, np.inf, True, "42", [42], 1j):
            with self.assertRaises(ValueError):
                aug.add_heteroscedastic_noise_in_normalised_luminosity(
                    [42], a=.01, b=.0004, x_ref=bad)
        for bad in (42, np.random.RandomState(42), object()):
            for call in self.calls([42], bad):
                with self.assertRaises(TypeError):
                    call()

    def test_zero_amplitude_is_exact_and_does_not_consume_rng(self):
        values = np.array([0., 42., 1000.])
        rng = np.random.default_rng(21)
        state = copy.deepcopy(rng.bit_generator.state)
        results = (
            aug.add_iid_gaussian_noise_in_log10_luminosity(values, sigma_dex=0, rng=rng),
            aug.add_heteroscedastic_noise_in_normalised_luminosity(values, a=0, b=0, rng=rng),
        )
        for result in results:
            np.testing.assert_array_equal(result, values)
            self.assertFalse(np.shares_memory(result, values))
        self.assertEqual(state, rng.bit_generator.state)

    def test_seeded_repeatability_including_frequent_resampling(self):
        values = np.linspace(0, 44, 1000).reshape(10, 100)
        first = self.calls(values, np.random.default_rng(20))
        second = self.calls(values, np.random.default_rng(20))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left(), right())

    def test_global_random_state_is_untouched(self):
        state = np.random.get_state()
        try:
            np.random.seed(105)
            expected = np.random.random(3)
            for rng in (None, np.random.default_rng(12)):
                np.random.seed(105)
                for call in self.calls([0., 39., 42.], rng):
                    call()
                np.testing.assert_array_equal(np.random.random(3), expected)
        finally:
            np.random.set_state(state)

    def test_dex_mean_variance_and_luminosity_mean(self):
        values = np.full(200_000, 42.)
        result = aug.add_iid_gaussian_noise_in_log10_luminosity(
            values, sigma_dex=.05, rng=np.random.default_rng(41))
        residual = result-values
        self.assertAlmostEqual(residual.mean(), 0, delta=.0005)
        self.assertAlmostEqual(residual.var(), .05**2, delta=.00004)
        expected = math.exp((math.log(10)*.05)**2/2)
        self.assertAlmostEqual(np.mean(10**residual), expected, delta=.001)

    def test_dex_independence_across_epochs_and_curves(self):
        result = aug.add_iid_gaussian_noise_in_log10_luminosity(
            np.full((6000, 8), 42.), sigma_dex=.05, rng=np.random.default_rng(15))
        self.assertLess(np.max(np.abs(np.corrcoef(result.T)-np.eye(8))), .06)
        self.assertLess(abs(np.corrcoef(result[:-1].ravel(), result[1:].ravel())[0, 1]), .03)
        single = aug.add_iid_gaussian_noise_in_log10_luminosity(
            np.full(100, 42.), sigma_dex=.05, rng=np.random.default_rng(15))
        self.assertGreater(np.std(single), .03)

    def test_heteroscedastic_untruncated_variance_law(self):
        for f in (1., 10., 100.):
            x = 42+math.log10(f)
            result = aug.add_heteroscedastic_noise_in_normalised_luminosity(
                np.full(120_000, x), a=.01, b=.0004, rng=np.random.default_rng(11))
            linear = 10**(result-42)
            variance = .01*f+.0004
            self.assertAlmostEqual(linear.mean(), f, delta=.015*math.sqrt(variance))
            self.assertAlmostEqual(linear.var(), variance, delta=.02*variance)

    def test_truncated_mean_and_variance_at_low_signal(self):
        f = .01
        scale = math.sqrt(.01*f+.0004)
        alpha = -f/scale
        survival = .5*math.erfc(alpha/math.sqrt(2))
        lam = math.exp(-alpha**2/2)/math.sqrt(2*math.pi)/survival
        expected_mean = f+scale*lam
        expected_var = scale**2*(1+alpha*lam-lam**2)
        result = aug.add_heteroscedastic_noise_in_normalised_luminosity(
            np.full(200_000, 40.), a=.01, b=.0004, rng=np.random.default_rng(8))
        linear = 10**(result-42)
        self.assertTrue(np.all(linear > 0))
        self.assertAlmostEqual(linear.mean(), expected_mean, delta=.01*expected_mean)
        self.assertAlmostEqual(linear.var(), expected_var, delta=.02*expected_var)
        self.assertGreater(np.unique(linear).size, 199_000)
        self.assertGreater(linear.mean(), 2*f)

    def test_zero_log_value_has_half_normal_limit(self):
        result = aug.add_heteroscedastic_noise_in_normalised_luminosity(
            np.zeros(100_000), a=.01, b=.0004, rng=np.random.default_rng(16))
        linear = 10**(result-42)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(linear.mean(), .02*math.sqrt(2/math.pi), delta=.0002)

    def test_source_only_and_constant_only_noise(self):
        for a, b in ((.01, 0), (0, .0004)):
            result = aug.add_heteroscedastic_noise_in_normalised_luminosity(
                np.full(100_000, 42.), a=a, b=b, rng=np.random.default_rng(42))
            linear = 10**(result-42)
            self.assertAlmostEqual(linear.var(), a+b, delta=.02*(a+b))

    def test_reference_is_a_rescalable_convention(self):
        values = np.linspace(40, 44, 1000)
        first = aug.add_heteroscedastic_noise_in_normalised_luminosity(
            values, a=.01, b=.0004, x_ref=42, rng=np.random.default_rng(3))
        # Increasing x_ref by one divides f by 10, a by 10 and b by 100.
        second = aug.add_heteroscedastic_noise_in_normalised_luminosity(
            values, a=.001, b=.000004, x_ref=43, rng=np.random.default_rng(3))
        np.testing.assert_allclose(first, second, atol=1e-12, rtol=0)

    def test_only_rejected_elements_are_resampled_from_original(self):
        rng = ScriptedGenerator([[-2., 1., -1.], [0., 2.]])
        result = aug.add_heteroscedastic_noise_in_normalised_luminosity(
            [42., 42., 42.], a=0, b=1, rng=rng)
        np.testing.assert_allclose(10**(result-42), [1., 2., 3.])
        self.assertEqual(rng.sizes, [3, 2])

    def test_resampling_limit_fails_without_clipping(self):
        rng = ScriptedGenerator([-2.] * 128)
        with self.assertRaisesRegex(RuntimeError, "128"):
            aug.add_heteroscedastic_noise_in_normalised_luminosity(
                [42.], a=0, b=1, rng=rng)
        self.assertEqual(len(rng.sizes), 128)

    def test_normalisation_overflow_and_underflow_fail(self):
        for x in (-400., 400.):
            with self.assertRaisesRegex(ValueError, "Normalised luminosity"):
                aug.add_heteroscedastic_noise_in_normalised_luminosity(
                    [x], a=.01, b=.0004, x_ref=0)

    def test_variance_overflow_and_underflow_fail(self):
        for x, a in ((200., 1e200), (-200., 1e-200)):
            with self.assertRaisesRegex(ValueError, "Noise variance"):
                aug.add_heteroscedastic_noise_in_normalised_luminosity(
                    [x], a=a, b=0, x_ref=0)

    def test_nonfinite_proposals_fail(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            aug.add_iid_gaussian_noise_in_log10_luminosity(
                [1e308], sigma_dex=1e308, rng=ScriptedGenerator([2.]))
        with self.assertRaisesRegex(ValueError, "not finite"):
            aug.add_heteroscedastic_noise_in_normalised_luminosity(
                [42.], a=.01, b=.0004, rng=ScriptedGenerator([np.inf]))

    def test_historical_functions_remain_numerically_unchanged(self):
        values = np.log(np.array([[4., 9., 16.], [25., 36., 49.]]))
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            slow = aug.add_gaussian_noise_slow(values, .05, rng=np.random.default_rng(12))
            fast = aug.add_gaussian_noise(values, .05, rng=np.random.default_rng(12))
            legacy, width = aug.add_exp_gaussian_log_noise(values, sigma=.05, random_state=12)
        self.assertEqual(emitted, [])
        np.testing.assert_array_equal(slow, values+.05*np.random.default_rng(12).standard_normal(values.shape))
        expected_fast = np.tile(np.random.default_rng(12).standard_normal(2), 3).reshape(2, 3)
        np.testing.assert_array_equal(fast, values+.05*expected_fast)
        linear = np.exp(values)
        std = .05*np.sqrt(linear)
        expected = np.log(np.maximum(linear+np.random.RandomState(12).normal(0, std, size=values.shape), 1e-12))
        np.testing.assert_array_equal(legacy, expected)
        np.testing.assert_array_equal(width, np.log(linear+std)-np.log(linear-std))

    def test_pipeline_does_not_call_new_kernels(self):
        with (
            patch.object(aug, "add_iid_gaussian_noise_in_log10_luminosity", side_effect=AssertionError),
            patch.object(aug, "add_heteroscedastic_noise_in_normalised_luminosity", side_effect=AssertionError),
        ):
            curves, mask = aug.apply_lsst_pipeline(
                np.full((2, 30), 42.), 30, .05, samples_per_day=1,
                rng=np.random.default_rng(5))
        self.assertEqual(curves.shape, mask.shape)
        self.assertTrue(np.isfinite(curves).all())


if __name__ == "__main__":
    unittest.main()
