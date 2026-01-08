"""Build a tabular dataset from cached NASA NeoWs JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_project.config import Paths
from my_project.logger import logger


@dataclass(frozen=True)
class Dataset:
    """Dataset bundle."""

    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _extract_row(obj: dict[str, Any]) -> dict[str, Any]:
    """Extract one asteroid into a flat row of features."""
    diam = obj.get("estimated_diameter", {}).get("kilometers", {})
    dmin = _safe_float(diam.get("estimated_diameter_min"))
    dmax = _safe_float(diam.get("estimated_diameter_max"))
    dmean = np.nanmean([dmin, dmax]).item() if not (np.isnan(dmin) and np.isnan(dmax)) else float("nan")

    cad = obj.get("close_approach_data", []) or []
    # Take the minimum miss distance (km) across approaches, and the maximum rel velocity.
    miss_km = float("nan")
    vel_kph = float("nan")
    earth_approaches = 0
    for ca in cad:
        md = _safe_float((ca.get("miss_distance") or {}).get("kilometers"))
        rv = _safe_float((ca.get("relative_velocity") or {}).get("kilometers_per_hour"))
        if not np.isnan(md):
            miss_km = md if np.isnan(miss_km) else min(miss_km, md)
        if not np.isnan(rv):
            vel_kph = rv if np.isnan(vel_kph) else max(vel_kph, rv)
        if (ca.get("orbiting_body") or "").upper() == "EARTH":
            earth_approaches += 1

    return {
        "id": str(obj.get("id", "")),
        "name": str(obj.get("name", "")),
        "absolute_magnitude_h": _safe_float(obj.get("absolute_magnitude_h")),
        "diam_km_min": dmin,
        "diam_km_max": dmax,
        "diam_km_mean": dmean,
        "min_miss_distance_km": miss_km,
        "max_relative_velocity_kph": vel_kph,
        "n_close_approaches": int(len(cad)),
        "n_earth_close_approaches": int(earth_approaches),
        "is_potentially_hazardous_asteroid": int(bool(obj.get("is_potentially_hazardous_asteroid", False))),
    }


def load_cached_objects(json_files: list[Path]) -> list[dict[str, Any]]:
    """Load cached pages and return a flat list of asteroid objects."""
    objects: list[dict[str, Any]] = []
    for fp in json_files:
        payload = json.loads(fp.read_text(encoding="utf-8"))
        page_objects = payload.get("near_earth_objects", []) or []
        objects.extend(page_objects)
    logger.info("Loaded %d raw objects from %d cached pages.", len(objects), len(json_files))
    return objects


def build_dataset(json_files: list[Path]) -> Dataset:
    """Build a dataset DataFrame from cached JSON files."""
    objs = load_cached_objects(json_files)
    rows = [_extract_row(o) for o in objs]
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Basic cleaning: drop rows with missing target (shouldn't happen) and remove all-empty feature rows.
    target = "is_potentially_hazardous_asteroid"
    df = df[df[target].isin([0, 1])].copy()

    feature_cols = [
        "absolute_magnitude_h",
        "diam_km_min",
        "diam_km_max",
        "diam_km_mean",
        "min_miss_distance_km",
        "max_relative_velocity_kph",
        "n_close_approaches",
        "n_earth_close_approaches",
    ]
    # Ensure numeric
    for c in feature_cols + [target]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    logger.info("Dataset shape: %s (hazardous rate=%.3f)", df.shape, df[target].mean())
    return Dataset(df=df, feature_cols=feature_cols, target_col=target)


def save_dataset_csv(dataset: Dataset, out_csv: Path) -> None:
    """Save dataset to CSV."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.df.to_csv(out_csv, index=False)
    logger.info("Saved dataset CSV: %s", out_csv)


def build_from_default_cache() -> Dataset:
    """Convenience: load all cached browse pages in data/raw."""
    paths = Paths.from_here()
    files = sorted(paths.data_raw.glob("neows_browse_page_*.json"))
    if not files:
        msg = "No cached files found in data/raw. Run: python -m my_project.fetch"
        raise FileNotFoundError(msg)
    return build_dataset(files)
