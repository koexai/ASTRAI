"""Numerical contracts for physical-time phenomenological masking."""
from dataclasses import replace
import unittest

import numpy as np

from astrai.utils.masking import (
    MaskingConfig, build_time_axis, generate_masking, generate_masking_realisation,
    interpolate_observations, resolve_samples_per_day, view_configuration,
)
from astrai.utils.lsst import random_cloud_masking, plot_masking_components


CLEAR = dict(daylight_mean_hours=0, daylight_amplitude_hours=0,
             seasonal_gap_max_days=0, moon_loss_hours=0, cloudy_fraction=0)


class MaskingTests(unittest.TestCase):
    def test_approved_defaults_and_effective_identity(self):
        c = MaskingConfig()
        self.assertEqual((c.cloudy_fraction, c.mean_cloudy_days, c.threshold_interval_days), (.3, 2.3, 1))
        self.assertEqual(c.record()['parameters']['seasonal_gap_max_days'], 90)
        self.assertEqual(view_configuration({}), view_configuration({'augmentation': {'masking': {}}}))
        self.assertEqual(view_configuration({})['samples_per_day'], 1)

    def test_invalid_parameters_are_rejected(self):
        for values in [dict(unknown=1), dict(cloudy_fraction=True), dict(cloudy_fraction=1.1),
                       dict(mean_cloudy_days=0), dict(threshold_interval_days=-1),
                       dict(daylight_amplitude_hours=13), dict(daylight_mean_hours=25),
                       dict(seasonal_gap_min_days=100), dict(seasonal_gap_max_days=366),
                       dict(moon_active_days=30), dict(moon_loss_hours=11),
                       dict(moon_period_days=float('nan')), dict(annual_period_days=float('inf'))]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                MaskingConfig.from_mapping(values)
        with self.assertRaises(ValueError):
            MaskingConfig.from_mapping([])

    def test_time_contract_and_invalid_grids(self):
        np.testing.assert_array_equal(build_time_axis(5, 4), [0, .25, .5, .75, 1])
        self.assertEqual(build_time_axis(421, 1)[-1], 420)
        self.assertEqual(build_time_axis(1601, 4)[-1], 400)
        for rate in (0, -1, True, float('inf'), '4'):
            with self.assertRaises(ValueError):
                resolve_samples_per_day({'data': {'samples_per_day': rate}})
        for size in (0, True, 1.5):
            with self.assertRaises(ValueError):
                build_time_axis(size)
        r = generate_masking_realisation(2, rng=np.random.default_rng(1))
        for times in ([], [-1, 0], [0, 3], [0, 0], [1, 0], [float('nan')]):
            with self.assertRaises(ValueError):
                r.evaluate(times)

    def test_daily_thresholds_match_the_worked_example(self):
        r = generate_masking_realisation(2, MaskingConfig(**{**CLEAR, 'daylight_mean_hours': 12}),
                                        np.random.default_rng(1))
        r = replace(r, threshold_offset=.3, thresholds=np.array([.2, .8, .3]))
        fine = r.evaluate(build_time_axis(9, 4))
        np.testing.assert_array_equal(fine['retained_mask'], [1, 1, 1, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(r.evaluate(build_time_axis(3, 1))['retained_mask'],
                                      fine['retained_mask'][::4])

    def test_threshold_boundary_is_left_closed(self):
        r = generate_masking_realisation(2, MaskingConfig(**CLEAR), np.random.default_rng(1))
        r = replace(r, threshold_offset=.25, thresholds=np.array([.1, .8, .2]))
        actual = r.evaluate([.749, .75, 1.749, 1.75])['threshold']
        np.testing.assert_array_equal(actual, [.1, .8, .8, .2])

    def test_same_path_agrees_on_nested_grids(self):
        for horizon in (400, 420):
            for seed in (0, 42, 123):
                r = generate_masking_realisation(horizon, rng=np.random.default_rng(seed))
                coarse = r.evaluate(build_time_axis(horizon + 1, 1))
                fine = r.evaluate(build_time_axis(horizon * 4 + 1, 4))
                for key in coarse:
                    np.testing.assert_array_equal(coarse[key], fine[key][::4])

    def test_regeneration_and_rng_consumption_do_not_depend_on_grid_size(self):
        first, second = np.random.default_rng(42), np.random.default_rng(42)
        coarse = generate_masking(build_time_axis(421, 1), rng=first)
        fine = generate_masking(build_time_axis(1681, 4), rng=second)
        np.testing.assert_array_equal(coarse['retained_mask'], fine['retained_mask'][::4])
        self.assertEqual(first.random(), second.random())

    def test_explicit_rng_does_not_change_global_state(self):
        np.random.seed(42)
        expected = np.random.random()
        np.random.seed(42)
        generate_masking(build_time_axis(421), rng=np.random.default_rng(12))
        self.assertEqual(np.random.random(), expected)

    def test_daylight_has_the_declared_annual_range(self):
        r = generate_masking_realisation(365.25, rng=np.random.default_rng(1))
        r = replace(r, daylight_phase=0)
        h = r.evaluate(np.array([0, .25, .5, .75, 1]) * 365.25)['daylight_hours']
        np.testing.assert_allclose(h, [12, 14, 12, 10, 12])

    def test_seasonal_gap_has_physical_duration_and_no_array_wrap(self):
        r = generate_masking_realisation(420, rng=np.random.default_rng(1))
        r = replace(r, seasonal_centre=0, seasonal_duration=10)
        t = np.array([0, 4.9, 5, 100, 360.25, 365.25, 370.25, 420])
        np.testing.assert_array_equal(r.evaluate(t)['sun_blocked'], [1, 1, 0, 0, 1, 1, 0, 0])

    def test_moon_profile_is_periodic_and_localised(self):
        r = generate_masking_realisation(60, rng=np.random.default_rng(1))
        r = replace(r, moon_phase=0)
        t = np.array([0, 2, 10, 29.53, 31.53, 39.53])
        moon = r.evaluate(t)['moon_contribution']
        np.testing.assert_allclose(moon[:3], moon[3:])
        self.assertEqual(moon[0], 1)
        self.assertEqual(moon[2], 0)

    def test_combined_availability_preserves_the_hours_budget(self):
        c = MaskingConfig(seasonal_gap_max_days=0, cloudy_fraction=0)
        parts = generate_masking(build_time_axis(421), c, np.random.default_rng(3))
        expected = (24 - parts['daylight_hours'] - 4 * parts['moon_contribution']) / 24
        np.testing.assert_allclose(parts['availability'], expected)
        np.testing.assert_array_equal(parts['retained_mask'], parts['threshold'] < expected)

    def test_sun_and_cloud_exclusions_cannot_be_reactivated(self):
        parts = generate_masking(build_time_axis(1681, 4), rng=np.random.default_rng(42))
        blocked = parts['sun_blocked'] | parts['cloud_blocked']
        self.assertTrue(blocked.any())
        self.assertFalse(parts['retained_mask'][blocked].any())
        self.assertTrue(np.all(parts['availability'][blocked] == 0))

    def test_clear_and_fully_cloudy_limits(self):
        t = build_time_axis(12)
        clear = generate_masking(t, CLEAR, np.random.default_rng(1))
        cloudy = generate_masking(t, {**CLEAR, 'cloudy_fraction': 1}, np.random.default_rng(1))
        self.assertTrue(clear['retained_mask'].all())
        self.assertFalse(cloudy['retained_mask'].any())
        np.testing.assert_array_equal(random_cloud_masking(np.ones(12), percentage=0, seed=1), np.ones(12))
        np.testing.assert_array_equal(random_cloud_masking(np.ones(12), percentage=100, seed=1), np.zeros(12))

    def test_cloud_transitions_have_explicit_boundary_semantics(self):
        r = generate_masking_realisation(10, rng=np.random.default_rng(1))
        r = replace(r, initially_cloudy=True, cloud_transitions=np.array([2.3, 7.0]))
        np.testing.assert_array_equal(r.evaluate([0, 2.299, 2.3, 6.9, 7, 10])['cloud_blocked'],
                                      [1, 1, 0, 0, 1, 1])

    def test_stationary_weather_occupancy_and_episode_means(self):
        r = generate_masking_realisation(50000, rng=np.random.default_rng(42))
        edges = np.r_[0, r.cloud_transitions, r.horizon_days]
        lengths = np.diff(edges)
        cloudy = np.logical_xor(r.initially_cloudy, np.arange(len(lengths)) % 2 == 1)
        self.assertAlmostEqual(np.sum(lengths[cloudy]) / r.horizon_days, .3, delta=.015)
        # Exclude both boundary-censored episodes when estimating spell means.
        inner, state = lengths[1:-1], cloudy[1:-1]
        self.assertAlmostEqual(inner[state].mean(), 2.3, delta=.12)
        self.assertAlmostEqual(inner[~state].mean(), 2.3*.7/.3, delta=.25)
        self.assertGreater(np.std(inner[state]), 1)

    def test_initial_cloud_probability_is_stationary(self):
        rng = np.random.default_rng(42)
        states = [generate_masking_realisation(0, rng=rng).initially_cloudy for _ in range(2000)]
        self.assertAlmostEqual(np.mean(states), .3, delta=.04)

    def test_pointwise_probability_matches_availability(self):
        rng = np.random.default_rng(42)
        c = MaskingConfig(**{**CLEAR, 'daylight_mean_hours': 12})
        kept = [generate_masking([0.], c, rng)['retained_mask'][0] for _ in range(2000)]
        self.assertAlmostEqual(np.mean(kept), .5, delta=.04)

    def test_cloud_helper_agrees_across_resolutions(self):
        a = random_cloud_masking(np.ones(421), seed=42, samples_per_day=1)
        b = random_cloud_masking(np.ones(1681), seed=42, samples_per_day=4)
        np.testing.assert_array_equal(a, b[::4])
        with self.assertRaisesRegex(ValueError, 'cannot be supplied together'):
            random_cloud_masking(np.ones(3), seed=1, rng=np.random.default_rng(1))

    def test_interpolation_uses_only_observations_and_constant_edges(self):
        a = interpolate_observations([np.nan, 40, np.nan, 44, np.nan],
                                      np.array([0, 1, 0, 1, 0], bool), np.arange(5))
        np.testing.assert_array_equal(a, [40, 40, 42, 44, 44])

    def test_interpolation_handles_single_and_zero_observations(self):
        mask = np.array([0, 1, 0], bool)
        np.testing.assert_array_equal(interpolate_observations([99, 42, -99], mask, [0, 1, 2]), [42]*3)
        with self.assertRaisesRegex(ValueError, 'No observations'):
            interpolate_observations([1, 2, 3], np.zeros(3, bool), [0, 1, 2])

    def test_interpolation_respects_nonuniform_times(self):
        a = interpolate_observations([40, 99, 44], np.array([1, 0, 1], bool), [0, 1, 4])
        np.testing.assert_array_equal(a, [40, 41, 44])

    def test_interpolation_rejects_invalid_observations(self):
        with self.assertRaises(ValueError):
            interpolate_observations([np.nan, 2], np.ones(2, bool), [0, 1])
        with self.assertRaises(ValueError):
            interpolate_observations([1, 2], np.ones(2), [0, 1])
        with self.assertRaises(ValueError):
            interpolate_observations([1, 2], np.ones(2, bool), [0, 0])

    def test_diagnostic_panels_use_actual_retained_points(self):
        import matplotlib.pyplot as plt
        parts = generate_masking(build_time_axis(421), rng=np.random.default_rng(42))
        figure = plot_masking_components(parts)
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(figure.axes), 7)
        observed = parts['time_days'][parts['retained_mask']]
        np.testing.assert_array_equal(figure.axes[5].lines[0].get_xdata(), observed)
        np.testing.assert_array_equal(figure.axes[6].lines[0].get_ydata(), np.diff(observed))

    def test_view_identity_captures_changed_masking_and_policy(self):
        baseline = view_configuration({'augmentation': {'noise_std': .05}})
        changed = view_configuration({'augmentation': {'noise_std': .05, 'masking': {'cloudy_fraction': .2}}})
        self.assertNotEqual(baseline, changed)
        self.assertEqual(baseline['masking']['interpolation']['zero_observations'], 'error')


if __name__ == '__main__':
    unittest.main()
