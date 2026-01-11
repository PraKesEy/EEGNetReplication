"""Build a tabular dataset from cached NASA NeoWs JSON."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import mne

from eegnet_repl.config import Paths
from eegnet_repl.logger import logger


@dataclass(frozen=True)
class Dataset:
    """Dataset bundle."""

    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str



'''
    Dataset build code for EEG data from raw data
'''

def exponential_moving_standardize(x: np.ndarray, factor_new: float = 0.001, init_block_size: int = 1000) -> np.ndarray:
    """Apply exponential moving standardization to the data.

    Args:
        x: Input data array of shape (n_channels, n_times).
        factor_new: Smoothing factor for the moving average.
        init_block_size: Number of initial samples to use for mean and std calculation.

    Returns:
        Standardized data array of the same shape as input.
    """
    x_std = np.zeros_like(x)
    mean_prev = np.mean(x[:, :init_block_size], axis=1, keepdims=True)
    var_prev = np.var(x[:, :init_block_size], axis=1, keepdims=True)

    for t in range(x.shape[1]):
        sample = x[:, t:t+1]
        mean_curr = (1 - factor_new) * mean_prev + factor_new * sample
        var_curr = (1 - factor_new) * var_prev + factor_new * (sample - mean_curr) ** 2

        x_std[:, t:t+1] = (sample - mean_curr) / np.sqrt(var_curr + 1e-10)

        mean_prev = mean_curr
        var_prev = var_curr

    return x_std

def preprocess_raw_data(src_path: Path, dest_path: Path) -> None:
    # Read raw EEG data files from src_path/Train
    for file in (src_path / "Train").glob("*.gdf"):
        raw = mne.io.read_raw_gdf(file, preload=True)

    # Rename the channels to more readable names
    channel_names = [
        'Fz',  'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5',  'C3',  'C1',  'Cz', 
        'C2',  'C4',  'C6',  'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1',  'Pz', 
        'P2',  'POz', 'EOG-left', 'EOG-central', 'EOG-right'
    ]

    mapping = {old_name: new_name for old_name, new_name in zip(raw.ch_names, channel_names)}
    raw.rename_channels(mapping)

    # Set EEG channel types
    eeg_channel_names = channel_names[:-3]  # All except the last three EOG
    raw.set_channel_types({ch_name: 'eeg' for ch_name in eeg_channel_names})
    # Set EOG channel types
    eog_channel_names = channel_names[-3:]  # Last three channels
    raw.set_channel_types({ch_name: 'eog' for ch_name in eog_channel_names})

    # Set sensor location information
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # Remove EOG channels
    eog_channels = ['EOG-left', 'EOG-central', 'EOG-right']
    raw.drop_channels(eog_channels)

    # Resample to 128 Hz
    raw.resample(128, npad="auto")

    # Adding a bandpass filter from 4Hz to 38Hz
    raw.filter(4., 38., fir_design='firwin', skip_by_annotation='edge')

    # Exponential moving standardization
    # exponential_moving_standardize(x, factor_new=0.001, init_block_size=1000)
    dataT = raw.get_data()
    dataT_std = exponential_moving_standardize(dataT)
    raw_std = raw.copy()
    raw_std._data = dataT_std
    raw_std.plot()

    # Save preprocessed data to dest_path
    dest_path.mkdir(parents=True, exist_ok=True)
    out_file = dest_path / file.name.replace('.gdf', '-preprocessed.fif')
    raw_std.save(out_file, overwrite=True)
    logger.info(f"Saved preprocessed file to {out_file}")
     

def preprocess_moabb_data(src_path: Path, dest_path: Path) -> None:
    pass
     

def build_eeg_dataset(src='kaggle') -> None:
    """Build EEG dataset from raw data files."""
    paths = Paths.from_here()
    if src == 'kaggle':
        raw_data_path = paths.data_raw
    elif src == 'moabb':
            raw_data_path = paths.data_moabb
    processed_data_path = paths.data_processed
    processed_data_path.mkdir(parents=True, exist_ok=True)

    # Preprocess raw EEG data files and build dataset
    if src == 'kaggle':
        preprocess_raw_data(raw_data_path, processed_data_path)
    elif src == 'moabb':
        preprocess_moabb_data(raw_data_path, processed_data_path)
    

