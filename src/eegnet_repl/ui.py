"""Tiny UI to explore asteroid hazard predictions."""

from __future__ import annotations

import pickle
import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
import pandas as pd

from eegnet_repl.config import Paths
from eegnet_repl.logger import logger
from eegnet_repl.model import TrainedModel


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


def main() -> None:
    """Run the UI."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
