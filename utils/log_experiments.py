"""
This module creates a timestamped experiment folder and saves
the source code and config so that every run is reproducible.
"""
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


_EXCLUDED_CODE_DIRS = {
    "__MACOSX",
    "__pycache__",
    "data",
    "experiments",
    "preprocessed",
    "venv",
}


def _iter_python_files(folder):
    """Yield Python files and their paths relative to the source root."""
    root = Path(folder).resolve()

    for current_dir, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not name.startswith(".")
            and name not in _EXCLUDED_CODE_DIRS
        )

        current_path = Path(current_dir)
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                file_path = current_path / filename
                yield file_path, file_path.relative_to(root).as_posix()


def create_experiment_dir(base_dir="experiments"):
    """Create a timestamped experiment directory.
    base_dir: the parent directory where experiment folders will be created
    Returns the path to the created experiment directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_code(exp_dir, folder="."):
    """Save Python source files recursively into the experiment archive.

    Files retain their paths relative to ``folder``. Generated data,
    experiment outputs, virtual environments, caches, and hidden directories
    are excluded.
    """
    zip_path = Path(exp_dir) / "code.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path, archive_path in _iter_python_files(folder):
            zipf.write(file_path, archive_path)

    print(f"Code saved to {zip_path}.")


def save_config(exp_dir, config_path="configs/default.yaml"):
    """Copy the config file into the experiment directory.
    exp_dir: the experiment directory where the config file will be saved
    config_path: the path to the config file to be copied
    """
    dst = os.path.join(exp_dir, os.path.basename(config_path))
    shutil.copy2(config_path, dst)
    print(f"Config saved to {dst}.")
