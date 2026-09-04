import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_characterizer
import train_generator


class LegacyTrainingArrayTests(unittest.TestCase):
    @staticmethod
    def _save_arrays(directory, names):
        for position, name in enumerate(names):
            values = np.full((2, 2), position + 0.25, dtype=np.float64)
            np.save(Path(directory) / name, values)

    def test_characterizer_loader_normalises_legacy_float64_arrays(self):
        names = (
            "x_train_clean_pca.npy",
            "x_train_aug_pca.npy",
            "x_test_pca.npy",
            "y_train_scaled.npy",
            "y_test.npy",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            self._save_arrays(temp_dir, names)

            arrays = train_characterizer._load_fold_data(temp_dir)

        self.assertTrue(
            all(array.dtype == np.dtype(np.float32) for array in arrays)
        )

    def test_generator_loader_normalises_legacy_float64_arrays(self):
        names = (
            "x_train_clean_pca.npy",
            "x_train_aug_pca.npy",
            "y_train_scaled.npy",
            "y_test_scaled.npy",
            "x_test_clean.npy",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            self._save_arrays(temp_dir, names)

            arrays = train_generator._load_fold_data(temp_dir)

        self.assertTrue(
            all(array.dtype == np.dtype(np.float32) for array in arrays)
        )


if __name__ == "__main__":
    unittest.main()
