from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from my_project.dataset import build_dataset


def _write_page(tmp_path: Path, objs: list[dict]) -> Path:
    payload = {"near_earth_objects": objs}
    fp = tmp_path / "page.json"
    fp.write_text(json.dumps(payload), encoding="utf-8")
    return fp


def test_build_dataset_smoke(tmp_path: Path) -> None:
    objs = [
        {
            "id": "1",
            "name": "Test A",
            "absolute_magnitude_h": 22.1,
            "estimated_diameter": {"kilometers": {"estimated_diameter_min": 0.1, "estimated_diameter_max": 0.2}},
            "close_approach_data": [
                {
                    "relative_velocity": {"kilometers_per_hour": "12345"},
                    "miss_distance": {"kilometers": "999999"},
                    "orbiting_body": "Earth",
                }
            ],
            "is_potentially_hazardous_asteroid": True,
        }
    ]
    fp = _write_page(tmp_path, objs)
    ds = build_dataset([fp])
    assert isinstance(ds.df, pd.DataFrame)
    assert ds.df.shape[0] == 1
    assert ds.target_col in ds.df.columns
    assert set(ds.feature_cols).issubset(ds.df.columns)
