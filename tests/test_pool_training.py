"""Small end-to-end smoke tests for pool-specific training and reload."""
import copy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import torch
import yaml

from astrai.cli import preprocess, train, train_characterizer, train_generator
from astrai.utils.checkpoints import load_characterizer, load_generator, load_unified_model
from astrai.utils.preprocessing import load_training_source, load_training_fold
from astrai.utils.target_transformations import validate_parameter_scalers


class PoolTrainingSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(14)
        self.curves = rng.uniform(1, 3, size=(18, 12)).astype(np.float32)
        self.parameters = rng.uniform(0.1, 2, size=(18, 2)).astype(np.float32)
        model = {"width": 4, "depth": 1, "dropout": 0.0}
        training = {"epochs": 2, "batch_size": 4, "learning_rate": 0.001, "test_fold": 1}
        checkpoint = {"model": "model.pth", "x_scaler": "x.pkl", "y_scaler": "y.pkl", "pca": "pca.pkl"}
        self.cfg = {"data": {"n_days": 12, "samples_per_day": 1, "n_params": 2,
                             "param_names": ["Mass", "Energy"]},
                    "preprocessing": {"n_splits": 3, "random_seed": 42, "pca_components": 2},
                    "partitioning": {"validation_fraction": 0.2, "selection_folds": 3},
                    "augmentation": {"noise_std": 0.05},
                    **{stage: {"model": copy.deepcopy(model), "training": copy.deepcopy(training),
                               "checkpoint": copy.deepcopy(checkpoint)}
                       for stage in ("characterizer", "generator")}}

    def prepare(self, cfg=None, name="prep"):
        with (patch.object(preprocess, "load_raw_data", return_value=(self.curves, self.parameters)),
              redirect_stdout(StringIO())):
            return Path(preprocess.run_preprocessing(cfg or self.cfg, self.root / name))

    def run_pair(self, prep, cfg=None):
        cfg = cfg or self.cfg
        with redirect_stdout(StringIO()):
            # Reversed order demonstrates that the partition is owned by neither stage.
            gen = train_generator.run_generator_training(cfg, prep, self.root / "gen")
            char = train_characterizer.run_characterizer_training(cfg, prep, self.root / "char")
        return Path(char), Path(gen)

    @staticmethod
    def metadata(path):
        return yaml.safe_load((path / "metadata.yaml").read_text())

    def test_single_fold_pair_shares_partitions_and_reloads(self):
        prep = self.prepare()
        char, gen = self.run_pair(prep)
        char_meta, gen_meta = self.metadata(char), self.metadata(gen)
        for role in ("training", "validation", "test"):
            np.testing.assert_array_equal(np.load(char / "fold_1" / f"{role}_indices.npy"),
                                          np.load(gen / "fold_1" / f"{role}_indices.npy"))
        self.assertEqual(char_meta["checkpoint"]["selected_checkpoint"]["preprocessing_bundle_id"],
                         gen_meta["checkpoint"]["selected_checkpoint"]["preprocessing_bundle_id"])
        cm, _, cy, _ = load_characterizer(self.cfg, torch.device("cpu"), char)
        gm, _, gy, _ = load_generator(self.cfg, torch.device("cpu"), gen)
        validate_parameter_scalers(cy, gy)
        source = load_training_source(prep, self.cfg)
        part, bundle = load_training_fold(prep, self.cfg, 1, source)
        test = part.indices("test")
        char_metrics = train_characterizer._evaluate_characterizer(
            cm, bundle.transform_curves(self.curves[test]), source["parameters"][test],
            self.cfg["data"]["param_names"], char, torch.device("cpu"), self.cfg, y_scaler=cy)
        self.assertAlmostEqual(char_metrics["transformed"]["aggregate"]["R2"],
                               char_meta["results"]["folds"][0]["metrics"]["test"]["transformed"]["aggregate"]["R2"])
        self.assertEqual(gm.training, False)

    def test_multifold_checkpoints_keep_their_actual_preprocessing(self):
        cfg = copy.deepcopy(self.cfg)
        for stage in ("characterizer", "generator"):
            cfg[stage]["training"]["test_fold"] = None
        prep = self.prepare(cfg)
        char, gen = self.run_pair(prep, cfg)
        for path, loader in ((char, load_characterizer), (gen, load_generator)):
            metadata = self.metadata(path)
            self.assertEqual(len(metadata["results"]["folds"]), 3)
            fold = metadata["checkpoint"]["selected_checkpoint"]["outer_fold"]
            _, _, scaler, _ = loader(cfg, torch.device("cpu"), path)
            np.testing.assert_array_equal(scaler.mean_, joblib.load(prep / f"fold_{fold}" / "y_scaler.pkl").mean_)

    def test_unified_training_records_pool_and_reloads(self):
        cfg = {"data": self.cfg["data"], "partitioning": self.cfg["partitioning"],
               "augmentation": self.cfg["augmentation"],
               "model": {**self.cfg["characterizer"]["model"], "pca_components": 2},
               "training": {**self.cfg["characterizer"]["training"], "n_splits": 3, "random_seed": 42},
               "checkpoint": self.cfg["characterizer"]["checkpoint"],
               "loss": {"alpha_char": 1.0, "alpha_gen": 1.0}}
        with (patch.object(train, "load_raw_data", return_value=(self.curves, self.parameters)),
              redirect_stdout(StringIO())):
            path = Path(train.run_unified_training(cfg, self.root / "unified"))
        _, _, scaler, _ = load_unified_model(cfg, torch.device("cpu"), path)
        metadata = self.metadata(path)
        fold = metadata["checkpoint"]["selected_checkpoint"]["outer_fold"]
        rows = np.load(path / f"fold_{fold}" / "training_indices.npy")
        np.testing.assert_allclose(scaler.mean_, np.log1p(self.parameters[rows]).mean(axis=0), atol=1e-7)
        self.assertTrue((path / "x_raw.npy").is_file())
        self.assertTrue((path / "partitions.yaml").is_file())

    def test_changed_augmentation_does_not_change_clean_fit(self):
        first = self.prepare()
        changed = copy.deepcopy(self.cfg)
        changed["augmentation"]["noise_std"] = 2.0
        second = self.prepare(changed, "other")
        first_source, second_source = load_training_source(first, self.cfg), load_training_source(second, changed)
        _, left = load_training_fold(first, self.cfg, 1, first_source)
        _, right = load_training_fold(second, changed, 1, second_source)
        self.assertEqual(left.manifest["learned_state"], right.manifest["learned_state"])
        self.assertFalse(np.array_equal(np.load(first / "fold_1/x_train_aug_pca.npy"),
                                       np.load(second / "fold_1/x_train_aug_pca.npy")))

    def test_stale_fold_arrays_are_rejected_before_training(self):
        prep = self.prepare()
        file = prep / "fold_1/y_train_scaled.npy"
        np.save(file, np.load(file) + 1)
        with (patch.object(train_generator, "build_generator") as build,
              redirect_stdout(StringIO())):
            with self.assertRaisesRegex(ValueError, "manifest"):
                train_generator.run_generator_training(self.cfg, prep, self.root / "failed")
        build.assert_not_called()

    def test_diagnostics_accept_the_matching_fold_bundle(self):
        from astrai.utils import plot_results
        prep = self.prepare()
        char, gen = self.run_pair(prep)
        output = self.root / "plots"
        with redirect_stdout(StringIO()):
            plot_results.main(["--exp-char", str(char), "--exp-gen", str(gen),
                               "--prep", str(prep), "--fold", "1", "--sample", "0",
                               "--output-dir", str(output)])
        self.assertTrue(list(output.glob("*.pdf")))


if __name__ == "__main__":
    unittest.main()
