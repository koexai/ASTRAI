"""Pairing requires provenance even when parameter scalers are numerically equal."""

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from astrai.utils.model_pairing import validate_cascade_pair
from astrai.utils.partitions import Partition
from astrai.utils.preprocessing import fit_preprocessing


class ModelPairingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.char, self.gen = self.root / "char", self.root / "gen"
        self.char.mkdir()
        self.gen.mkdir()
        rng = np.random.default_rng(14)
        self.curves = rng.normal(size=(12, 6)).astype(np.float32)
        self.parameters = rng.uniform(size=(12, 2)).astype(np.float32)
        self.partition = Partition(12, tuple(range(10)), tuple(range(8)),
                                   (8, 9), (10, 11), outer_fold=1)
        self.cfg = {"data": {"n_params": 2, "param_names": ["Mass", "Energy"]}}
        self.bundle = self.fit()
        self.write(self.char, "characterizer", self.bundle, epoch=1)
        self.write(self.gen, "generator", self.bundle, epoch=3)

    def fit(self, *, partition=None, cfg=None, data_id=None):
        return fit_preprocessing(self.curves, self.parameters, partition or self.partition,
                                 2, 42, cfg or self.cfg, data_id=data_id)

    @staticmethod
    def write(path, stage, bundle, epoch=1):
        metadata = {
            "experiment_metadata_version": 7,
            "run": {"stage": stage, "status": "completed"},
            "checkpoint": {"selected_checkpoint": {
                "outer_fold": bundle.manifest["partition"]["outer_fold"],
                "role": "holdout_validation", "epoch": epoch,
                "preprocessing_bundle_id": bundle.manifest["bundle_id"],
            }},
            "results": {"folds": [{
                "outer_fold": bundle.manifest["partition"]["outer_fold"],
                "preprocessing": bundle.manifest,
            }]},
        }
        (path / "metadata.yaml").write_text(yaml.safe_dump(metadata))

    def validate(self, gen_scaler=None):
        validate_cascade_pair(self.char, self.gen, self.bundle.y_scaler,
                              self.bundle.y_scaler if gen_scaler is None else gen_scaler)

    def mutate_metadata(self, change):
        path = self.gen / "metadata.yaml"
        metadata = yaml.safe_load(path.read_text())
        change(metadata)
        path.write_text(yaml.safe_dump(metadata))

    def test_same_bundle_accepts_independently_selected_epochs(self):
        self.validate(copy.deepcopy(self.bundle.y_scaler))

    def test_cross_fold_pair_is_rejected_even_with_equal_scalers(self):
        partition = replace(self.partition, outer_fold=2)
        other = self.fit(partition=partition)
        np.testing.assert_array_equal(other.y_scaler.mean_, self.bundle.y_scaler.mean_)
        self.write(self.gen, "generator", other)
        with self.assertRaisesRegex(ValueError, "same outer fold"):
            self.validate(other.y_scaler)

    def test_different_dataset_is_rejected_even_with_equal_scalers(self):
        other = self.fit(data_id="another-dataset")
        self.write(self.gen, "generator", other)
        with self.assertRaisesRegex(ValueError, "same outer fold"):
            self.validate(other.y_scaler)

    def test_different_training_pool_is_rejected(self):
        partition = Partition(12, tuple(range(10)), tuple(range(2, 10)),
                              (0, 1), (10, 11), outer_fold=1)
        other = self.fit(partition=partition)
        self.write(self.gen, "generator", other)
        with self.assertRaisesRegex(ValueError, "same outer fold"):
            self.validate(other.y_scaler)

    def test_different_parameter_names_or_units_are_rejected(self):
        for change in ({"param_names": ["Energy", "Mass"]}, {"param_units": ["g", "erg"]}):
            with self.subTest(change=change):
                cfg = copy.deepcopy(self.cfg)
                cfg["data"].update(change)
                other = self.fit(cfg=cfg)
                self.write(self.gen, "generator", other)
                with self.assertRaisesRegex(ValueError, "same outer fold"):
                    self.validate(other.y_scaler)

    def test_unverifiable_historical_metadata_is_rejected(self):
        for version in (None, 4):
            with self.subTest(version=version):
                self.mutate_metadata(lambda m: m.update(experiment_metadata_version=version))
                with self.assertRaisesRegex(ValueError, "historical pairs without this provenance"):
                    self.validate()
        (self.gen / "metadata.yaml").unlink()
        with self.assertRaisesRegex(ValueError, "historical pairs without this provenance"):
            self.validate()

    def test_unassociated_historical_scaler_is_rejected(self):
        scaler = copy.deepcopy(self.bundle.y_scaler)
        del scaler.astrai_association
        with self.assertRaisesRegex(ValueError, "scaler association differs"):
            self.validate(scaler)

    def test_changed_transform_association_is_rejected(self):
        scaler = copy.deepcopy(self.bundle.y_scaler)
        scaler.astrai_association["target_transform"] = {"name": "identity"}
        with self.assertRaisesRegex(ValueError, "target_transform"):
            self.validate(scaler)

    def test_changed_scaler_state_is_rejected(self):
        scaler = copy.deepcopy(self.bundle.y_scaler)
        scaler.mean_[0] += 1
        with self.assertRaisesRegex(ValueError, "scaler state differs"):
            self.validate(scaler)

    def test_incomplete_run_or_wrong_stage_is_rejected(self):
        for change in ({"status": "failed"}, {"stage": "characterizer"}):
            with self.subTest(change=change):
                self.write(self.gen, "generator", self.bundle)
                self.mutate_metadata(lambda m: m["run"].update(change))
                with self.assertRaisesRegex(ValueError, "completed generator"):
                    self.validate()

    def test_selected_checkpoint_must_identify_the_recorded_fold_and_bundle(self):
        for change in ({"outer_fold": 2}, {"preprocessing_bundle_id": "other"},
                       {"role": "outer_refit"}):
            with self.subTest(change=change):
                self.write(self.gen, "generator", self.bundle)
                self.mutate_metadata(lambda m: m["checkpoint"]["selected_checkpoint"].update(change))
                with self.assertRaisesRegex(ValueError, "Cannot verify cascade pairing"):
                    self.validate()

    def test_tampered_bundle_manifest_is_rejected(self):
        self.mutate_metadata(lambda m: m["results"]["folds"][0]["preprocessing"].update(dataset_id="tampered"))
        with self.assertRaisesRegex(ValueError, "Invalid selected preprocessing identity"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
