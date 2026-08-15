import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from femto_rul.features.prefix import build_prefix_training_samples
from femto_rul.models.prefix_models import TotalLifeToRULRegressor


def _bearing(name: str, n: int = 101) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "condition": 1,
                "bearing": name,
                "file_index": i,
                "rul_seconds": (n - 1 - i) * 10.0,
                "rms_horiz": 1.0 + i * 0.01,
                "rms_vert": 1.2 + i * 0.01,
                "kurtosis_horiz": 2.0 + i * 0.02,
                "kurtosis_vert": 2.1 + i * 0.02,
                "crest_factor_horiz": 3.0 + i * 0.005,
                "crest_factor_vert": 3.1 + i * 0.005,
            }
        )
    return pd.DataFrame(rows)


def test_total_life_is_constant_across_prefixes_of_complete_training_bearing():
    out = build_prefix_training_samples(_bearing("Bearing1_1"), fractions=(0.4, 0.6, 0.8))
    assert out["total_life_seconds"].nunique() == 1
    assert float(out["total_life_seconds"].iloc[0]) == 1000.0


def test_total_life_wrapper_converts_predicted_life_back_to_rul():
    X = pd.DataFrame(
        {
            "observed_age_seconds": [100.0, 200.0],
            "x": [1.0, 2.0],
        }
    )
    y = np.array([900.0, 800.0])
    model = TotalLifeToRULRegressor(
        DummyRegressor(strategy="constant", constant=1000.0)
    ).fit(X, y)
    pred = model.predict(X)
    assert np.allclose(pred, [900.0, 800.0])
