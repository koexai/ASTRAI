"""Noise identities must be explicit, unambiguous and serialisable."""
import unittest

import yaml

from astrai.utils.noise_configuration import NoiseConfig, resolve_noise_config


class NoiseConfigurationTests(unittest.TestCase):
    def test_five_models_and_resolved_defaults(self):
        cases = [
            {"model": "iid_log10", "sigma_dex": .05},
            {"model": "heteroscedastic_normalised", "a": .01, "b": .0004},
            {"model": "legacy_iid_gaussian", "noise_std": .05},
            {"model": "legacy_tiled_gaussian", "noise_std": .05},
            {"model": "legacy_exp_sqrt_gaussian", "sigma": 1},
        ]
        records = []
        for mapping in cases:
            with self.subTest(model=mapping["model"]):
                config = resolve_noise_config({"augmentation": {"noise": mapping}})
                records.append(config.record())
                self.assertEqual(yaml.safe_load(yaml.safe_dump(config.record())), config.record())
                explicit = {"model": config.model, **dict(config.parameters)}
                self.assertEqual(config, NoiseConfig.from_mapping(explicit))
        self.assertEqual(records[1]["parameters"]["x_ref"], 42.)
        self.assertEqual(records[-1]["parameters"]["eps"], 1e-12)
        self.assertEqual(len({r["model"] for r in records}), 5)

    def test_missing_old_and_ambiguous_contracts_fail(self):
        for cfg in [{}, {"augmentation": {}}, {"augmentation": None},
                    {"augmentation": {"noise_std": .05}},
                    {"augmentation": {"noise_std": .05, "noise": {
                        "model": "iid_log10", "sigma_dex": .05}}}]:
            with self.subTest(cfg=cfg), self.assertRaisesRegex(ValueError, "augmentation.noise"):
                resolve_noise_config(cfg)

    def test_invalid_models_parameters_and_scalars_fail(self):
        cases = [None, [], {}, {"model": "unknown"}, {"model": []},
                 {"model": "iid_log10"},
                 {"model": "iid_log10", "sigma_dex": .05, "a": 0},
                 {"model": "heteroscedastic_normalised", "a": 1},
                 {"model": "legacy_exp_sqrt_gaussian", "sigma": 1, "eps": 0}]
        for bad in [None, True, "0.05", [], -1, float("nan"), float("inf")]:
            cases.append({"model": "iid_log10", "sigma_dex": bad})
        for mapping in cases:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                NoiseConfig.from_mapping(mapping)

    def test_zero_rule_is_fixed_metadata_not_a_user_option(self):
        mapping = {"model": "heteroscedastic_normalised", "a": 0, "b": 0}
        record = NoiseConfig.from_mapping(mapping).record()
        self.assertFalse(record["zero_treatment_scientifically_validated"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            NoiseConfig.from_mapping({**mapping, "zero_treatment": "missing"})


if __name__ == "__main__":
    unittest.main()
