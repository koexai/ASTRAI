"""Unified diagnostics describe the exact new corruption of each shown sample."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yaml

from astrai.utils import visualize_reconstruction as diagnostic
from astrai.utils.augmentation import apply_augmentation
from astrai.utils.augmentation_configuration import view_configuration
from astrai.utils.reproducibility import derive_diagnostic_seed


class ReconstructionNoiseProvenanceTests(unittest.TestCase):
    def test_command_exports_sample_local_recipe_and_actual_plotted_values(self):
        curves = np.random.default_rng(13).uniform(40, 43, (4, 421)).astype(np.float32)
        curves[:, :3] = 0
        x_scaler = StandardScaler().fit(curves)
        pca = PCA(n_components=2).fit(x_scaler.transform(curves))
        y_scaler = StandardScaler().fit([[1.], [2.], [3.], [4.]])
        for noise in [{"model": "iid_log10", "sigma_dex": .05},
                      {"model": "heteroscedastic_normalised", "a": .01, "b": .0004}]:
            with self.subTest(noise=noise), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cfg = {"data": {"n_days": 421, "param_names": ["Mass"], "samples_per_day": 1},
                       "augmentation": {"noise": noise}}
                config_file = root / "config.yaml"
                config_file.write_text(yaml.safe_dump(cfg))
                with (patch.object(diagnostic, "load_from_experiment", return_value=(None, x_scaler, y_scaler, pca)),
                      patch.object(diagnostic, "load_raw_data", return_value=(curves, None)),
                      patch.object(diagnostic, "reconstruct_all", return_value=(curves, np.zeros((4, 1)))),
                      patch.object(diagnostic.plt, "show"), redirect_stdout(StringIO())):
                    diagnostic.main(["--config", str(config_file), "--exp", str(root / "experiment"),
                                     "--index", "2", "--lsst-seed", "17", "--output-dir", str(root / "plots")])
                record = json.loads((root / "plots/augmentation_metadata_2.json").read_text())
                self.assertEqual(record["view_configuration"], view_configuration(cfg))
                self.assertEqual(record["sample_seed"], derive_diagnostic_seed(17, 2))
                self.assertEqual(record["augmentation"]["excluded_zero_count"],
                                 3 if noise["model"] == "heteroscedastic_normalised" else 0)
                self.assertEqual(record["purpose"], "new_diagnostic_corruption")
                self.assertFalse(record["historical_mask_reproduction"])
                expected, _ = apply_augmentation(curves[2:3], 421, noise, seed=record["sample_seed"])
                np.testing.assert_array_equal(plt.gcf().axes[0].lines[1].get_ydata(), expected[0])
                self.assertTrue((root / "plots/reconstruction_2.pdf").read_bytes().startswith(b"%PDF"))
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
