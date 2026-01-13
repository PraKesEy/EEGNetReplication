"""Build a tabular dataset from cached NASA NeoWs JSON."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mne
from numpy import multiply

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
    create_windows_from_events
)
from eegnet_repl.config import Paths
from eegnet_repl.logger import logger


@dataclass(frozen=True)
class BCICI2ADataset(Dataset):
    """Dataset bundle for BCI Competition IV Dataset 2a."""

    X: np.ndarray  # Shape: (n_samples, n_channels, n_times)
    y: np.ndarray  # Shape: (n_samples,)

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """Return a single sample and its label."""
        return self.X[idx], int(self.y[idx])

def raw_exponential_moving_standardize(x: np.ndarray, factor_new: float = 0.001, init_block_size: int = 1000) -> np.ndarray:
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
    '''
    Preprocess raw EEG data files from src_path and save to dest_path.
    
    Args:
        src_path: Path to the directory containing raw EEG data files.
        dest_path: Path to the directory to save preprocessed EEG data files.

    Returns:
        None
    '''
    logger.info(f"Preprocessing raw data from {src_path} to {dest_path}")
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
        dataT_std = raw_exponential_moving_standardize(dataT)
        raw_std = raw.copy()
        raw_std._data = dataT_std

        # Save preprocessed data to dest_path
        dest_path.mkdir(parents=True, exist_ok=True)
        out_file = dest_path / file.name.replace('.gdf', '-preprocessed.fif')
        raw_std.save(out_file, overwrite=True)
        logger.info(f"Saved preprocessed file to {out_file}")

def build_dataset_from_preprocessed(src='kaggle', subject='all') -> BCICI2ADataset:
    '''
    Build a Dataset object from preprocessed EEG data files in dest_path.
    
    Args:
        src: Source of the dataset ('kaggle' or 'moabb').
        subject: Subject identifier [1-9] (default is 'all' to include all subjects).

    Returns:
        Dataset object containing the data and metadata.
    '''
    paths = Paths.from_here()
    if src == 'kaggle':
        dest_path = paths.data_processed
    elif src == 'moabb':
        dest_path = paths.data_moabb_processed
    else:
        raise ValueError(f"Unknown source: {src}")
    logger.info(f"Building dataset from preprocessed data in {dest_path}")

    if subject != 'all':
        # Filter files for the specified subject
        files = list(dest_path.glob(f"sub-{subject:02d}-*-preprocessed.fif"))
    else:
        # Include all preprocessed files
        files = list(dest_path.glob("*-preprocessed.fif"))
    if not files:
        raise ValueError(f"No preprocessed files found in {dest_path} for subject {subject}")
    logger.info(f"Found {len(files)} preprocessed files for subject {subject}")
    all_data = []
    all_labels = []
    for file in files:
        pp = mne.io.read_raw_fif(file, preload=True)
        # create a plot of annotation descriptions over time
        annotationsT = pp.annotations.description

        # Annotation conversion map
        annotation_map = {
            '276': 'Idling EEG (eyes open)',
            '277': 'Idling EEG (eyes closed)',
            '768': 'Start of a trial',
            '769': 'Cue onset left (class 1)',
            '770': 'Cue onset right (class 2)',
            '771': 'Cue onset foot (class 3)',
            '772': 'Cue onset tongue (class 4)',
            '783': 'Cue unknown',
            '1023': 'Rejected trial',
            '1072': 'Eye movements',
            '32766': 'Start of a new run',
        }

        # Map the annotations to their descriptions
        annotationsT = [annotation_map.get(desc, desc) for desc in annotationsT]
        eventsT, event_idT = mne.events_from_annotations(pp)

        # Replace event IDs with annotation descriptions
        event_idT = {annotation_map.get(str(key), str(key)): value for key, value in event_idT.items()}

        # Break the data into trial windows (0.5-2.5 seconds cue onset) using cue onset markers
        all_event_ids = {'Cue onset left (class 1)': 7,
                         'Cue onset right (class 2)': 8,
                         'Cue onset foot (class 3)': 9,
                         'Cue onset tongue (class 4)': 10}
        # Filter to only include events that exist in this file
        available_event_ids = {event_name: event_id for event_name, event_id in all_event_ids.items() 
                               if event_id in event_idT.values()}
        
        if not available_event_ids:
            logger.warning(f"No matching events found in {file}, skipping this file")
            continue
        
        epocsT_std = mne.Epochs(pp, eventsT, event_id=available_event_ids,
                                tmin=0.5, tmax=2.5, baseline=None, preload=True)
        
        all_data.append(epocsT_std.get_data())  # Shape: (n_epochs, n_channels, n_times)
        all_labels.append(epocsT_std.events[:, -1])  # Extract labels from events

    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return BCICI2ADataset(X=X, y=y)
        
     

def preprocess_moabb_data(src_path: Path, dest_path: Path) -> None:
    # Read raw EEG data files from src_path/Train
    for file in (src_path / "Train").glob("*.fif"):
        raw = mne.io.read_raw_fif(file, preload=True)

        # Preprocessing steps
        low_cut_hz = 4.0  # low cut frequency for filtering
        high_cut_hz = 38.0  # high cut frequency for filtering
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000
        # Factor to convert from V to uV
        factor = 1e6
        #  New sampling rate
        new_sfreq = 128.0

        preprocessors = [
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
            Preprocessor("resample", sfreq=new_sfreq),  # Resample to new_sfreq
            Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
            Preprocessor(
                exponential_moving_standardize,  # Exponential moving standardization
                factor_new=factor_new,
                init_block_size=init_block_size,
            ),
        ]

        # Transform the data
        preprocess(raw, preprocessors, n_jobs=-1)
     

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


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Preprocess BCI Competition IV Dataset 2a from source.")
    parser.add_argument("--src", default="kaggle", help="Specify source (options: kaggle, moabb).")
    args = parser.parse_args()

    if args.src not in ['kaggle', 'moabb']:
        logger.error("Unknown source specified: %s", args.src)
        raise ValueError(f"Unknown source: {args.src}")
    
    paths = Paths.from_here()
    raw_data_path = paths.data_raw if args.src == 'kaggle' else paths.data_moabb
    processed_data_path = paths.data_processed if args.src == 'kaggle' else paths.data_moabb_processed

    logger.info("Preprocessing data from source: %s", args.src)
    if args.src == "kaggle":
        preprocess_raw_data(raw_data_path, processed_data_path)
    elif args.src == "moabb":
        preprocess_moabb_data(raw_data_path, processed_data_path)
    else:
        logger.error("Unknown source specified: %s", args.src)
        raise ValueError(f"Unknown source: {args.src}")


if __name__ == "__main__":
    main()