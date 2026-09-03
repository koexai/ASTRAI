import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils.plot_semi_analytical_curves import (
    build_time_axis,
    find_unique_sample,
    parse_parameter_selection,
    plot_one_at_a_time,
    plot_quantile_summary,
    plot_selected_curves,
    main as plot_main,
    resolve_one_at_a_time_groups,
    resolve_quantile_summary_groups,
    resolve_selected_indices,
    validate_sample_index,
)


PARAMETER_NAMES = ["Mass", "Energy"]
PARAMETERS = np.array(
    [
        [10.0, 1.0],
        [20.0, 1.0],
        [30.0, 1.0],
        [20.0, 2.0],
    ],
    dtype=np.float32,
)


class SemiAnalyticalSelectionTests(unittest.TestCase):
    def test_parses_parameter_selection(self):
        selection = parse_parameter_selection("Mass=20, Energy=1.5")

        self.assertEqual(selection, {"Mass": 20.0, "Energy": 1.5})

    def test_rejects_invalid_parameter_selection(self):
        with self.assertRaisesRegex(ValueError, "NAME=VALUE"):
            parse_parameter_selection("Mass:20")

        with self.assertRaisesRegex(ValueError, "more than once"):
            parse_parameter_selection("Mass=10,Mass=20")

    def test_finds_one_exact_parameter_match(self):
        index = find_unique_sample(
            PARAMETERS,
            PARAMETER_NAMES,
            {"Mass": 20.0, "Energy": 2.0},
        )

        self.assertEqual(index, 3)

    def test_rejects_zero_or_multiple_matches(self):
        with self.assertRaisesRegex(ValueError, "found 0"):
            find_unique_sample(
                PARAMETERS,
                PARAMETER_NAMES,
                {"Mass": 99.0},
            )

        with self.assertRaisesRegex(ValueError, "found 3"):
            find_unique_sample(
                PARAMETERS,
                PARAMETER_NAMES,
                {"Energy": 1.0},
            )

    def test_rejects_unknown_parameter_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown parameter"):
            find_unique_sample(
                PARAMETERS,
                PARAMETER_NAMES,
                {"Radius": 10.0},
            )

    def test_validates_index_range_and_type(self):
        self.assertEqual(validate_sample_index(3, 4), 3)

        with self.assertRaisesRegex(ValueError, "outside the valid range"):
            validate_sample_index(4, 4)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            validate_sample_index(True, 4)

    def test_resolves_and_deduplicates_explicit_selections(self):
        indices = resolve_selected_indices(
            PARAMETERS,
            PARAMETER_NAMES,
            indices=[0, 3],
            selections=[{"Mass": 20.0, "Energy": 2.0}],
        )

        self.assertEqual(indices, [0, 3])

    def test_resolves_exact_one_at_a_time_groups(self):
        specification = {
            "reference": {"Mass": 20.0, "Energy": 1.0},
            "levels": {
                "Mass": [10.0, 20.0, 30.0],
                "Energy": [1.0, 2.0],
            },
            "titles": {"Mass": "Ejecta mass"},
        }

        groups = resolve_one_at_a_time_groups(
            PARAMETERS,
            PARAMETER_NAMES,
            specification,
        )

        self.assertEqual(groups[0]["parameter"], "Mass")
        self.assertEqual(groups[0]["title"], "Ejecta mass")
        self.assertEqual(
            groups[0]["rows"],
            [(10.0, 0), (20.0, 1), (30.0, 2)],
        )
        self.assertEqual(
            groups[1]["rows"],
            [(1.0, 1), (2.0, 3)],
        )

    def test_builds_time_axis_from_sampling_rate(self):
        np.testing.assert_allclose(
            build_time_axis(4, samples_per_day=2),
            [0.0, 0.5, 1.0, 1.5],
        )

    def test_builds_marginal_quantile_summary_groups(self):
        curves = np.arange(24, dtype=float).reshape(8, 3)
        parameters = np.column_stack(
            [
                np.arange(8, dtype=float),
                np.arange(8, dtype=float)[::-1],
            ]
        )
        specification = {
            "quantile_ranges": {
                "Lower": [0.0, 0.25],
                "Central": [0.375, 0.625],
                "Upper": [0.75, 1.0],
            }
        }

        groups = resolve_quantile_summary_groups(
            curves,
            parameters,
            PARAMETER_NAMES,
            specification,
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["parameter"], "Mass")
        self.assertEqual(len(groups[0]["summaries"]), 3)
        np.testing.assert_allclose(
            groups[0]["summaries"][0]["curve"],
            np.median(curves[:2], axis=0),
        )
        np.testing.assert_allclose(
            groups[1]["summaries"][0]["curve"],
            np.median(curves[-2:], axis=0),
        )

    def test_rejects_invalid_quantile_range(self):
        with self.assertRaisesRegex(ValueError, "lower < upper"):
            resolve_quantile_summary_groups(
                np.ones((4, 3)),
                PARAMETERS,
                PARAMETER_NAMES,
                {"quantile_ranges": {"Invalid": [0.8, 0.2]}},
            )


class SemiAnalyticalPlotTests(unittest.TestCase):
    def setUp(self):
        self.curves = np.array(
            [
                [42.0, 42.1, 42.2],
                [42.0, 42.2, 42.4],
                [42.0, 42.3, 42.6],
                [42.0, 42.4, 42.8],
            ],
            dtype=np.float32,
        )
        self.time = build_time_axis(3, samples_per_day=1)

    def test_saves_selected_curve_formats_headlessly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = plot_selected_curves(
                self.curves,
                PARAMETERS,
                PARAMETER_NAMES,
                [0, 3],
                self.time,
                tmp_dir,
                "selected",
                ["pdf", "png"],
            )

            self.assertEqual(
                {path.name for path in saved},
                {"selected.pdf", "selected.png"},
            )
            for path in saved:
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_saves_one_at_a_time_plot_headlessly(self):
        specification = {
            "reference": {"Mass": 20.0, "Energy": 1.0},
            "levels": {
                "Mass": [10.0, 20.0, 30.0],
                "Energy": [1.0, 2.0],
            },
        }
        groups = resolve_one_at_a_time_groups(
            PARAMETERS,
            PARAMETER_NAMES,
            specification,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = plot_one_at_a_time(
                self.curves,
                groups,
                self.time,
                tmp_dir,
                "one_at_a_time",
                ["png"],
            )

            self.assertEqual(len(saved), 1)
            self.assertGreater(saved[0].stat().st_size, 0)

    def test_cli_loads_and_plots_configured_npy_csv_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            np.save(root / "curves.npy", self.curves)
            pd.DataFrame(
                PARAMETERS,
                columns=PARAMETER_NAMES,
            ).to_csv(root / "params.csv", index=False)
            config = {
                "data": {
                    "format": "npy_csv",
                    "curves_path": "curves.npy",
                    "params_path": "params.csv",
                    "n_days": 3,
                    "n_params": 2,
                    "param_names": PARAMETER_NAMES,
                    "samples_per_day": 1,
                },
                "visualisation": {
                    "one_at_a_time": {
                        "reference": {"Mass": 20.0, "Energy": 1.0},
                        "levels": {
                            "Mass": [10.0, 20.0, 30.0],
                            "Energy": [1.0, 2.0],
                        },
                    }
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            output_dir = root / "plots"

            plot_main(
                [
                    "--config",
                    str(config_path),
                    "--data-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--format",
                    "png",
                ]
            )

            output_path = output_dir / "clean_light_curves_2par.png"
            self.assertGreater(output_path.stat().st_size, 0)

    def test_saves_quantile_summary_plot_headlessly(self):
        groups = resolve_quantile_summary_groups(
            self.curves,
            PARAMETERS,
            PARAMETER_NAMES,
            {
                "quantile_ranges": {
                    "Lower": [0.0, 0.25],
                    "Central": [0.25, 0.75],
                    "Upper": [0.75, 1.0],
                }
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = plot_quantile_summary(
                groups,
                self.time,
                tmp_dir,
                "quantile_summary",
                ["png"],
            )

            self.assertEqual(len(saved), 1)
            self.assertGreater(saved[0].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
