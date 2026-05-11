"""
This module creates a timestamped experiment folder and saves
the source code and config so that every run is reproducible.
"""
import zipfile
import os
import shutil
from datetime import datetime


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
    """Save all .py files into a zip archive inside the experiment directory.
    exp_dir: the experiment directory where code.zip will be saved
    folder: the folder to search for .py files (default is current directory)"""
    zip_path = os.path.join(exp_dir, "code.zip")
    files_to_zip = [f for f in os.listdir(folder) if f.endswith(".py")]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))

    print(f"Code saved to {zip_path}.")


def save_config(exp_dir, config_path="configs/default.yaml"):
    """Copy the config file into the experiment directory.
    exp_dir: the experiment directory where the config file will be saved
    config_path: the path to the config file to be copied
    """
    dst = os.path.join(exp_dir, os.path.basename(config_path))
    shutil.copy2(config_path, dst)
    print(f"Config saved to {dst}.")
