import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data import load_raw_data


def make_npy_csv_config(curves_path, params_path, n_days=3):
    return {
        "data": {
            "format": "npy_csv",
            "curves_path": str(curves_path),
            "params_path": str(params_path),
            "params_csv_sep": ";",
            "n_days": n_days,
            "n_params": 2,
            "param_names": ["Mass", "Energy"],
            "samples_per_day": 1,
        }
    }


class RawDataLoadingTests(unittest.TestCase):
    def test_loads_npy_csv_parameters_without_transforming_them(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            curves_path = root / "curves.npy"
            params_path = root / "params.csv"
            curves = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            parameters = np.array([[10.0, 1.0], [20.0, 2.0]])
            np.save(curves_path, curves)
            pd.DataFrame(
                parameters,
                columns=["Mass", "Energy"],
            ).to_csv(params_path, sep=";", index=False)

            loaded_curves, loaded_parameters = load_raw_data(
                None,
                make_npy_csv_config(curves_path, params_path),
            )

            np.testing.assert_allclose(loaded_curves, curves)
            np.testing.assert_allclose(loaded_parameters, parameters)
            self.assertEqual(loaded_curves.dtype, np.dtype("float32"))
            self.assertEqual(loaded_parameters.dtype, np.dtype("float32"))

    def test_existing_loader_keeps_log1p_parameter_contract(self):
        from utils.checkpoints import load_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            curves_path = root / "curves.npy"
            params_path = root / "params.csv"
            curves = np.array([[1.0, 2.0, 3.0]])
            parameters = np.array([[10.0, 1.0]])
            np.save(curves_path, curves)
            pd.DataFrame(
                parameters,
                columns=["Mass", "Energy"],
            ).to_csv(params_path, sep=";", index=False)

            loaded_curves, loaded_parameters = load_data(
                None,
                make_npy_csv_config(curves_path, params_path),
            )

            np.testing.assert_allclose(loaded_curves, curves)
            np.testing.assert_allclose(
                loaded_parameters,
                np.log1p(parameters),
            )

    def test_loads_parquet_with_configured_curve_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = Path(tmp_dir) / "dataset.parquet"
            frame = pd.DataFrame(
                {
                    "0": [1.0, 4.0],
                    "1": [2.0, 5.0],
                    "2": [3.0, 6.0],
                    "Mass": [10.0, 20.0],
                    "Energy": [1.0, 2.0],
                }
            )
            frame.to_parquet(parquet_path, index=False)
            cfg = {
                "data": {
                    "format": "parquet",
                    "path": str(parquet_path),
                    "n_days": 3,
                    "n_params": 2,
                    "param_names": ["Mass", "Energy"],
                    "samples_per_day": 1,
                }
            }

            curves, parameters = load_raw_data(None, cfg)

            np.testing.assert_allclose(
                curves,
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            )
            np.testing.assert_allclose(
                parameters,
                [[10.0, 1.0], [20.0, 2.0]],
            )

    def test_rejects_mismatched_npy_csv_sample_counts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            curves_path = root / "curves.npy"
            params_path = root / "params.csv"
            np.save(curves_path, np.ones((2, 3)))
            pd.DataFrame(
                [[10.0, 1.0]],
                columns=["Mass", "Energy"],
            ).to_csv(params_path, sep=";", index=False)

            with self.assertRaisesRegex(
                ValueError,
                "different sample counts",
            ):
                load_raw_data(
                    None,
                    make_npy_csv_config(curves_path, params_path),
                )

    def test_rejects_partially_present_parameter_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            curves_path = root / "curves.npy"
            params_path = root / "params.csv"
            np.save(curves_path, np.ones((1, 3)))
            pd.DataFrame({"Mass": [10.0]}).to_csv(
                params_path,
                sep=";",
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "missing columns: Energy"):
                load_raw_data(
                    None,
                    make_npy_csv_config(curves_path, params_path),
                )

    def test_rejects_unknown_data_format(self):
        cfg = {
            "data": {
                "format": "fits",
                "n_days": 3,
                "n_params": 2,
                "param_names": ["Mass", "Energy"],
            }
        }

        with self.assertRaisesRegex(ValueError, "Unsupported data format"):
            load_raw_data(None, cfg)


if __name__ == "__main__":
    unittest.main()
