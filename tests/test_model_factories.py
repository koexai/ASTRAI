import tempfile
import unittest
from pathlib import Path

import joblib
import torch

from astrai.models.factories import (
    build_characterizer,
    build_generator,
    build_unified_model,
)
from astrai.models.split_mlp import SplitMLPRegressor
from astrai.models.residual_blocks import MLPWithResiduals
from astrai.models.unified_model import UnifiedModel
from astrai.utils.checkpoints import (
    load_characterizer,
    load_generator,
    load_unified_model,
)


class MarkerArtefact:
    pass


class ModelFactoryTests(unittest.TestCase):
    def setUp(self):
        self.split_cfg = {
            "data": {"n_params": 2},
            "preprocessing": {"pca_components": 3},
            "characterizer": {
                "model": {"width": 4, "depth": 1, "dropout": 0.0},
                "checkpoint": {
                    "model": "characterizer.pth",
                    "x_scaler": "char_x.pkl",
                    "y_scaler": "char_y.pkl",
                    "pca": "char_pca.pkl",
                },
            },
            "generator": {
                "model": {"width": 5, "depth": 2, "dropout": 0.0},
                "checkpoint": {
                    "model": "generator.pth",
                    "x_scaler": "gen_x.pkl",
                    "y_scaler": "gen_y.pkl",
                    "pca": "gen_pca.pkl",
                },
            },
        }
        self.unified_cfg = {
            "data": {"n_params": 2},
            "model": {
                "pca_components": 3,
                "width": 4,
                "depth": 1,
                "dropout": 0.0,
            },
            "checkpoint": {
                "model": "unified.pth",
                "x_scaler": "unified_x.pkl",
                "y_scaler": "unified_y.pkl",
                "pca": "unified_pca.pkl",
            },
        }

    def test_factories_build_the_configured_architectures(self):
        characterizer = build_characterizer(self.split_cfg)
        generator = build_generator(self.split_cfg)
        unified = build_unified_model(self.unified_cfg)

        self.assertIsInstance(characterizer, SplitMLPRegressor)
        self.assertEqual(len(characterizer.nets), 2)
        self.assertIsInstance(generator, MLPWithResiduals)
        self.assertEqual(generator.network[0].in_features, 2)
        self.assertEqual(generator.network[-1].out_features, 3)
        self.assertIsInstance(unified, UnifiedModel)
        self.assertEqual(len(unified.regressor.nets), 2)
        self.assertEqual(unified.generator.network[-1].out_features, 3)

    @staticmethod
    def _write_bundle(directory, checkpoint_cfg, model):
        torch.save(model.state_dict(), directory / checkpoint_cfg["model"])
        for key in ("x_scaler", "y_scaler", "pca"):
            joblib.dump(MarkerArtefact(), directory / checkpoint_cfg[key])

    @staticmethod
    def _assert_same_state(expected, actual):
        expected_state = expected.state_dict()
        actual_state = actual.state_dict()
        assert expected_state.keys() == actual_state.keys()
        for name in expected_state:
            torch.testing.assert_close(
                expected_state[name],
                actual_state[name],
                rtol=0,
                atol=0,
            )

    def test_split_checkpoint_loaders_reconstruct_factory_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            characterizer = build_characterizer(self.split_cfg)
            generator = build_generator(self.split_cfg)
            self._write_bundle(
                root,
                self.split_cfg["characterizer"]["checkpoint"],
                characterizer,
            )
            self._write_bundle(
                root,
                self.split_cfg["generator"]["checkpoint"],
                generator,
            )

            loaded_characterizer, *char_artefacts = load_characterizer(
                self.split_cfg,
                torch.device("cpu"),
                root,
            )
            loaded_generator, *gen_artefacts = load_generator(
                self.split_cfg,
                torch.device("cpu"),
                root,
            )

        self._assert_same_state(characterizer, loaded_characterizer)
        self._assert_same_state(generator, loaded_generator)
        self.assertTrue(all(isinstance(item, MarkerArtefact) for item in char_artefacts))
        self.assertTrue(all(isinstance(item, MarkerArtefact) for item in gen_artefacts))
        self.assertFalse(loaded_characterizer.training)
        self.assertFalse(loaded_generator.training)

    def test_unified_checkpoint_loader_reconstructs_factory_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = build_unified_model(self.unified_cfg)
            self._write_bundle(root, self.unified_cfg["checkpoint"], model)

            loaded, *artefacts = load_unified_model(
                self.unified_cfg,
                torch.device("cpu"),
                root,
            )

        self._assert_same_state(model, loaded)
        self.assertTrue(all(isinstance(item, MarkerArtefact) for item in artefacts))
        self.assertFalse(loaded.training)


if __name__ == "__main__":
    unittest.main()
