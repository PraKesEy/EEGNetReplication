"""Fetch BCI Competition IV Dataset 2a.

We use a kaggle dataset "prashastham/bci-competition-iv-dataset-2a".
Alternatively, we can fetch from moabb.
"""

from __future__ import annotations

import argparse
import time

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
    cache_path = kagglehub.dataset_download(dataset)

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
    from moabb.datasets import BNCI2014001

    paths = Paths.from_here()
    paths.data_raw.mkdir(parents=True, exist_ok=True)

    if dataset == MOABB_DATASET:
        dataset_obj = BNCI2014001()
    else:
        logger.error("Unknown moabb dataset specified: %s", dataset)
        raise ValueError(f"Unknown moabb dataset: {dataset}")

    # Fetch data for all subjects and save to data/moabb
    out_dir = paths.data_moabb
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train and Eval folders
    train_dir = out_dir / "Train"
    eval_dir = out_dir / "Eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for subject in dataset_obj.subject_list:
        logger.info("Fetching data for subject: %s", subject)
        subject_data = dataset_obj.get_data(subjects=[subject])[subject]

        # moabb returns a nested mapping {session: {run: Raw}}
        for session, runs in subject_data.items():
            for run_name, raw in runs.items():
                # If train data save to train folder, else eval folder
                if session == '0train':
                    out_dir = train_dir
                else:
                    out_dir = eval_dir
                out_path = out_dir / f"A0{subject}{'T' if session=='0train' else 'E'}_{run_name}.fif"
                raw.save(out_path, overwrite=True)
                logger.info(
                    "Saved subject=%s session=%s run=%s to %s", subject, session, run_name, out_path
                )
                time.sleep(1)  # Be polite to the server

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
