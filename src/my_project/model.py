"""Model definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainedModel:
    """Bundle for a trained model and metadata."""

    pipeline: Pipeline
    feature_cols: list[str]
    target_col: str

    def predict_proba_hazard(self, X: Any) -> np.ndarray:
        """Return probability of class=1 (hazardous)."""
        proba = self.pipeline.predict_proba(X)
        # assuming classes [0,1]
        return proba[:, 1]


def build_pipeline(seed: int) -> Pipeline:
    """Create a simple, strong baseline pipeline."""
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ],
    )
