"""Tiny UI to explore asteroid hazard predictions."""

from __future__ import annotations

import pickle
import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
import mne

from eegnet_repl.config import Paths
from eegnet_repl.logger import logger
#from eegnet_repl.model import TrainedModel


def load_model(path: Path) -> TrainedModel:
    """Load trained model."""
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, TrainedModel):
        raise TypeError("Loaded object is not a TrainedModel.")
    return obj


def _format_float(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.3g}"


class App(tk.Tk):
    """Asteroid explorer UI."""

    def __init__(self) -> None:
        super().__init__()
        self.title("NEO Hazard Lab — Asteroid Risk Explorer")
        self.geometry("980x520")

        paths = Paths.from_here()
        self.dataset_csv = paths.data_processed / "neo_dataset.csv"
        self.model_path = paths.models / "hazard_model.pkl"

        if not self.dataset_csv.exists() or not self.model_path.exists():
            msg = (
                "Missing data/model. Run:\n"
                "  python -m eegnet_repl.fetch --pages 3\n"
                "  python -m eegnet_repl.train\n"
            )
            tk.messagebox.showerror("Setup required", msg)  # type: ignore[attr-defined]
            raise FileNotFoundError(msg)

        self.df = pd.read_csv(self.dataset_csv)
        self.model = load_model(self.model_path)

        # Controls
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Button(top, text="Random sample (10)", command=self.show_random_sample).pack(side="left", padx=5)
        ttk.Button(top, text="Top risk (10)", command=self.show_top_risk).pack(side="left", padx=5)
        ttk.Button(top, text="Plot: diameter vs hazard", command=self.plot_diameter).pack(side="left", padx=5)

        self.status = ttk.Label(top, text="")
        self.status.pack(side="right")

        # Table
        cols = ("name", "diam_km_mean", "min_miss_distance_km", "max_relative_velocity_kph", "p_hazard", "true")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=18)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        headers = {
            "name": "Name",
            "diam_km_mean": "Diameter (km)",
            "min_miss_distance_km": "Min miss dist (km)",
            "max_relative_velocity_kph": "Max rel vel (kph)",
            "p_hazard": "Model P(hazard)",
            "true": "True hazard",
        }
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=140 if c != "name" else 260, anchor="w")

        self.show_random_sample()

    def _predict_for_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = rows[self.model.feature_cols]
        probs = self.model.predict_proba_hazard(X)
        out = rows.copy()
        out["p_hazard"] = probs
        return out

    def _set_rows(self, rows: pd.DataFrame, title: str) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for _, r in rows.iterrows():
            values = (
                str(r.get("name", "")),
                _format_float(r.get("diam_km_mean")),
                _format_float(r.get("min_miss_distance_km")),
                _format_float(r.get("max_relative_velocity_kph")),
                f"{float(r.get('p_hazard', 0.0)):.3f}",
                str(int(r.get("is_potentially_hazardous_asteroid", 0))),
            )
            self.tree.insert("", "end", values=values)

        rate = float(self.df["is_potentially_hazardous_asteroid"].mean())
        self.status.config(text=f"{title} | dataset hazard-rate={rate:.3f}")
        logger.info("UI view: %s (%d rows)", title, len(rows))

    def show_random_sample(self) -> None:
        n = min(10, len(self.df))
        idx = random.sample(range(len(self.df)), k=n)
        rows = self._predict_for_rows(self.df.iloc[idx])
        self._set_rows(rows, "Random sample")

    def show_top_risk(self) -> None:
        scored = self._predict_for_rows(self.df)
        top = scored.sort_values("p_hazard", ascending=False).head(10)
        self._set_rows(top, "Top model risk")

    def plot_diameter(self) -> None:
        # Simple scatter: diameter vs predicted probability
        scored = self._predict_for_rows(self.df)
        x = scored["diam_km_mean"].astype(float)
        y = scored["p_hazard"].astype(float)
        plt.figure()
        plt.scatter(x, y)
        plt.xlabel("Diameter (km)")
        plt.ylabel("Model P(hazard)")
        plt.title("Diameter vs predicted hazard probability")
        plt.show()

# Functions for EEGNet, not connected to UI yet

def plot_temporal_filters(model_dict) -> None:
    # Plot learned temporal filters
    temporal_filters = model_dict['temporal.0.weight']
    n_filters = temporal_filters.shape[0]
    plt.figure(figsize=(10, 6))
    for i in range(n_filters):
        plt.plot(temporal_filters[i, 0, 0].cpu().numpy(), label=f'Filter {i+1}')
    plt.title('Learned Temporal Filters')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def plot_spatial_filters(model_dict) -> None:
    # Visualize learned spatial filters using montages
    # Isolate the area that has electrodes and plot topomaps
    info = mne.create_info(
        ch_names=[
                'Fz',  'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5',  'C3',  'C1',  'Cz', 
                'C2',  'C4',  'C6',  'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1',  'Pz', 
                'P2',  'POz'
            ],
        sfreq=128,
        ch_types='eeg'
    )
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    spatial_filters = model_dict['spatial.weight']  # shape = (F1*D, F1, C, 1)
    num_filters = spatial_filters.shape[0]
    # Plot in a grid
    n_cols = 4
    n_rows = int(np.ceil(num_filters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    for i in range(num_filters):
        ax = axes[i // n_cols, i % n_cols]
        filter_weights = spatial_filters[i, 0, :, 0].cpu().numpy()  # shape = (C,)
        plot_topomap(filter_weights, info, axes=ax, show=False, cmap='viridis')
        ax.set_title(f'Spatial Filter {i+1}')
    plt.tight_layout()
    plt.show()

def PS(time_signal, f_sampling, method='ps'):
    fft = np.fft.fft(time_signal)
    mag_squared = np.real(fft * np.conjugate(fft))
    f = np.fft.fftfreq(len(time_signal), 1/f_sampling)

    if method == 'psd':
        scaling_factor = 2 / (f_sampling * len(time_signal))
    else:
        scaling_factor = 2 / (len(time_signal)**2) 

    PS = scaling_factor * mag_squared
    return f, PS

def plot_power_spectra_of_temporal_filters(model_dict) -> None:
    # Plotting learned temporal filters in subplots
    temporal_filters = model_dict['temporal.0.weight']
    n_filters = temporal_filters.shape[0]
    n_cols = 4
    n_rows = n_filters // n_cols + int(n_filters % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    for i in range(n_filters):
        row = i // n_cols
        col = i % n_cols
        #f, ps_version1 = PS(time_signal=temporal_filters[i, 0, 0].cpu().numpy(), f_sampling=128, method='psd') # Power Spectral Density scaling
        f, ps_version2 = PS(time_signal=temporal_filters[i, 0, 0].cpu().numpy(), f_sampling=128, method='ps')  # Power Spectrum scaling
        #axes[row, col].plot(f[0:len(f)//2-1], ps_version1[0:len(f)//2-1], 'ko-')
        axes[row, col].plot(f[0:len(f)//2-1], ps_version2[0:len(f)//2-1], 'ro-')
        axes[row, col].set_title(f'Temporal Filter {i+1}')
        axes[row, col].set_xlabel('Frequency (Hz)')
        axes[row, col].set_ylabel('Power (dB)')
        axes[row, col].set_xticks(range(0,51,10))
        #axes[row, col].legend(['PSD', 'Power Spectrum'])
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the UI."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
