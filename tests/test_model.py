from __future__ import annotations

import numpy as np
import pandas as pd

from eegnet_repl.model import TrainedModel, build_pipeline


def test_pipeline_predict_proba_shape() -> None:
    X = pd.DataFrame(
        {
            "absolute_magnitude_h": [20.0, 25.0],
            "diam_km_min": [0.1, 0.01],
            "diam_km_max": [0.2, 0.02],
            "diam_km_mean": [0.15, 0.015],
            "min_miss_distance_km": [1e6, 5e6],
            "max_relative_velocity_kph": [20000, 15000],
            "n_close_approaches": [1, 0],
            "n_earth_close_approaches": [1, 0],
        }
    )
    y = np.array([1, 0], dtype=int)
    pipe = build_pipeline(seed=0)
    pipe.fit(X, y)

    tm = TrainedModel(pipeline=pipe, feature_cols=list(X.columns), target_col="is_potentially_hazardous_asteroid")
    p = tm.predict_proba_hazard(X)
    assert p.shape == (2,)
    assert np.all((p >= 0.0) & (p <= 1.0))
