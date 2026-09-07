"""Configuration loading helpers shared by ASTRAI command-line workflows."""

from pathlib import Path

import yaml


def load_config(path):
    """Load a YAML configuration from a user or package resource path."""
    config_path = Path(path).expanduser()
    with config_path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)
