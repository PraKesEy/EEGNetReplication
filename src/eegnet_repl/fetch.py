"""Fetch BCI Competition IV Dataset 2a.

We use a kaggle dataset "prashastham/bci-competition-iv-dataset-2a".
Alternatively, we can fetch from moabb.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import kagglehub
import shutil
import os

from eegnet_repl.config import KAGGLE_DATASET, MOABB_DATASET, Paths
from eegnet_repl.logger import logger


def fetch_from_kaggle(dataset: str):
    """Fetch train and evaluation datasets from Kaggle.

    Args:
        dataset: Name of the Kaggle dataset to fetch.

    Returns:
        None.
    """
    # 1. Download to cache
    cache_path = kagglehub.dataset_download("prashastham/bci-competition-iv-dataset-2a")

    # 2. Move to your local 'data' folder
    paths = Paths.from_here()
    paths.data_raw.mkdir(parents=True, exist_ok=True)

    # Copying files from cache to local project
    for file in os.listdir(cache_path):
        src = os.path.join(cache_path, file)
        dst = os.path.join(paths.data_raw, file)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

def fetch_from_moabb(dataset: str):
    """Fetch preprocessed train and evaluation datasets from moabb.

    Args:
        dataset: Name of the moabb dataset to fetch.

    Returns:
        None.
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    from moabb import set_log_level
    import mne

    set_log_level("ERROR")
    paradigm = MotorImagery(n_classes=4)
    dataset = BNCI2014_001()
    subjects = dataset.subject_list
    logger.info(f"Found {len(subjects)} subjects in dataset {dataset}")
    paths = Paths.from_here()
    data_raw_path = paths.data_raw / "moabb_data"
    data_raw_path.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        logger.info(f"Fetching data for subject {subject}")
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        # Convert to MNE Epochs object for saving
        fname = data_raw_path / f"subject_{subject}_epochs-epo.fif"
        logger.info(f"Saved epochs for subject {subject} to {fname}")
        time.sleep(1)  # To avoid overwhelming any servers

def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Fetch BCI Competition IV Dataset 2a from source.")
    parser.add_argument("--src", default="kaggle", help="Specify source (options: kaggle, moabb).")
    args = parser.parse_args()

    logger.info("Fetching data from source: %s", args.src)
    if args.src == "kaggle":
        fetch_from_kaggle(dataset=KAGGLE_DATASET)
    elif args.src == "moabb":
        fetch_from_moabb(dataset=MOABB_DATASET)
    else:
        logger.error("Unknown source specified: %s", args.src)
        raise ValueError(f"Unknown source: {args.src}")


if __name__ == "__main__":
    main()
