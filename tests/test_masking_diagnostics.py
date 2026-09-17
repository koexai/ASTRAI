"""Diagnostic reports must describe the canonical masks without altering them."""
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import numpy as np
import yaml

from astrai.utils.masking import build_time_axis, generate_masking_realisation
from astrai.utils.masking_diagnostics import analyse, cloud_occupancy, main, observation_metrics


CLEAR = dict(daylight_mean_hours=0, daylight_amplitude_hours=0,
             seasonal_gap_max_days=0, moon_loss_hours=0, cloudy_fraction=0)


class MaskingDiagnosticTests(unittest.TestCase):
    def test_interpolation_metrics_separate_boundaries_and_missing_points(self):
        t = np.arange(5, dtype=float)
        y = np.array([40., 41., 43., 43., 44.])
        metrics, filled = observation_metrics(t, y, np.array([False, True, False, True, False]))
        np.testing.assert_array_equal(filled, [41, 41, 42, 43, 43])
        self.assertEqual(metrics['max_internal_gap_days'], 2)
        self.assertEqual(metrics['leading_gap_days'], 1)
        self.assertEqual(metrics['trailing_gap_days'], 1)
        self.assertEqual(metrics['rmse_missing_dex'], 1)
        self.assertAlmostEqual(metrics['rmse_all_dex'], np.sqrt(3 / 5))

    def test_empty_and_single_observation_are_reported_without_redraw(self):
        t = np.arange(5, dtype=float)
        y = t + 40
        metrics, filled = observation_metrics(t, y, np.zeros(5, dtype=bool))
        self.assertEqual(metrics['retained_count'], 0)
        self.assertTrue(np.isnan(filled).all())
        metrics, filled = observation_metrics(t, y, np.array([False, False, True, False, False]))
        np.testing.assert_array_equal(filled, np.full(5, 42))
        self.assertTrue(np.isnan(metrics['max_internal_gap_days']))
        self.assertEqual((metrics['leading_gap_days'], metrics['trailing_gap_days']), (2, 2))
        rows, _, example = analyse(t, y, n_realisations=4, config={'cloudy_fraction': 1})
        self.assertEqual([r['retained_count'] for r in rows], [0] * 4)
        self.assertTrue(np.isnan(example['filled']).all())

    def test_cloud_occupancy_uses_physical_durations_including_boundaries(self):
        path = generate_masking_realisation(10, rng=np.random.default_rng(42))
        path = replace(path, initially_cloudy=True, cloud_transitions=np.array([1.5, 8., 9.]))
        self.assertEqual(cloud_occupancy(path), .25)
        path = replace(path, initially_cloudy=False)
        self.assertEqual(cloud_occupancy(path), .75)

    def test_ensemble_preserves_paths_inputs_and_resolution_contract(self):
        t = build_time_axis(1601, 4)
        y = 42 + np.sin(t / 30)
        before = y.copy()
        rows, resolutions, example = analyse(t, y, seed=42, n_realisations=12)
        again = analyse(t, y, seed=42, n_realisations=12)
        self.assertEqual(rows, again[0])
        self.assertEqual(resolutions, again[1])
        np.testing.assert_array_equal(y, before)
        direct = generate_masking_realisation(400, rng=np.random.default_rng(42)).evaluate(t)
        np.testing.assert_array_equal(example['parts']['retained_mask'], direct['retained_mask'])
        for row in resolutions:
            self.assertEqual(row['common_time_mask_mismatches'], 0)
            self.assertEqual(row['common_time_max_availability_difference'], 0)
        for row in rows:
            fractions = [row[name + '_retained_fraction'] for name in
                         ('daylight', 'daylight_moon', 'daylight_moon_sun', 'all_components')]
            self.assertTrue(np.all(np.diff(fractions) <= 0))
            self.assertEqual(fractions[-1], row['retained_fraction'])

    def test_invalid_inputs_are_rejected(self):
        for t, y, kwargs in [([0], [42], {}), ([0, 0], [42, 43], {}),
                            ([1, 2], [42, 43], {}), ([0, 1], [42], {}),
                            ([0, 1], [42, np.nan], {}), ([0, 1], [42, 43], {'n_realisations': 0}),
                            ([0, 1], [42, 43], {'seed': -1})]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyse(t, y, **kwargs)

    def test_cli_exports_real_curve_config_and_reproducible_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            curves = np.array([[40, 41, 42, 43, 44], [44, 43, 42, 41, 40]], dtype=float)
            np.save(root / 'clean.npy', curves)
            (root / 'config.yaml').write_text(yaml.safe_dump({'augmentation': {'masking': CLEAR}}))
            args = ['--output-dir', str(root / 'report'), '--n-samples', '5', '--samples-per-day', '1',
                    '--n-realisations', '3', '--curve-file', str(root / 'clean.npy'), '--curve-index', '1',
                    '--config', str(root / 'config.yaml')]
            with redirect_stdout(StringIO()):
                summary = main(args)
            self.assertEqual(summary['mean_retained_fraction'], 1)
            self.assertEqual(summary['defined_missing_rmse_count'], 0)
            self.assertIsNone(summary['median_missing_rmse_dex'])
            self.assertEqual(summary['curve']['index'], 1)
            data = np.genfromtxt(root / 'report/example.csv', delimiter=',', names=True)
            np.testing.assert_array_equal(data['clean_log10_luminosity'], curves[1])
            np.testing.assert_array_equal(data['interpolated_log10_luminosity'], curves[1])
            self.assertEqual(json.loads((root / 'report/summary.json').read_text()), summary)
            for name in ('ensemble', 'interpolation', 'resolution'):
                self.assertTrue((root / f'report/{name}.pdf').read_bytes().startswith(b'%PDF'))
            for name in ('realisations', 'resolution'):
                self.assertEqual(len((root / f'report/{name}.csv').read_text().splitlines()), 4)
            with self.assertRaises(FileExistsError):
                main(args)

    def test_cli_exports_empty_masks_without_invalid_json_or_interpolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'clouds.yaml').write_text('augmentation:\n  masking:\n    cloudy_fraction: 1\n')
            with redirect_stdout(StringIO()):
                summary = main(['--output-dir', str(root / 'report'), '--n-samples', '5',
                                '--n-realisations', '2', '--config', str(root / 'clouds.yaml')])
            self.assertEqual(summary['zero_observation_count'], 2)
            self.assertIsNone(summary['median_missing_rmse_dex'])
            data = np.genfromtxt(root / 'report/example.csv', delimiter=',', names=True)
            self.assertTrue(np.isnan(data['interpolated_log10_luminosity']).all())
            self.assertFalse(data['retained_mask'].any())
            text = (root / 'report/summary.json').read_text()
            self.assertNotIn('NaN', text.replace('NaN when no observations', ''))


if __name__ == '__main__':
    unittest.main()
