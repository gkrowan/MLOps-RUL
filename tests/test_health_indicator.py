import numpy as np
import pandas as pd

from femto_rul.features.health_indicator import (
    build_health_indicator_samples,
    health_indicator_feature_columns,
)


def _bearing(name: str, condition: int, n: int = 180) -> pd.DataFrame:
    x = np.arange(n, dtype=float)
    data = {
        "condition": condition,
        "bearing": name,
        "file_index": np.arange(n),
        "rul_seconds": (n - 1 - np.arange(n)) * 10.0,
        "rms_horiz": 1.0 + 0.002 * x,
        "rms_vert": 1.2 + 0.001 * x,
        "kurtosis_horiz": 0.2 + 0.003 * x,
        "kurtosis_vert": 0.3 + 0.002 * x,
        "crest_factor_horiz": 3.0 + 0.001 * x,
        "crest_factor_vert": 3.1 + 0.0015 * x,
    }
    for i in range(8):
        data[f"fft_band_{i}_horiz"] = 1.0 + i + 0.001 * x
        data[f"fft_band_{i}_vert"] = 1.5 + i + 0.0015 * x
    return pd.DataFrame(data)


def test_health_indicator_contract_is_finite_and_target_free():
    train = pd.concat([_bearing("Bearing1_x", 1), _bearing("Bearing2_x", 2)], ignore_index=True)
    out = build_health_indicator_samples(train, fractions=(0.4, 0.6, 0.8), healthy_window=20, recent_window=20)
    assert len(out) == 6
    features = health_indicator_feature_columns()
    assert "rul_seconds" not in features
    assert np.isfinite(out[[*features, "rul_seconds"]].to_numpy(dtype=float)).all()


def test_health_indicator_is_causal_to_future_changes():
    base = _bearing("Bearing1_x", 1, n=200)
    fractions = (0.6,)
    first = build_health_indicator_samples(base, fractions=fractions, healthy_window=20, recent_window=20)

    cut = int(round((len(base) - 1) * 0.6))
    changed = base.copy()
    future = changed.index > cut
    for col in [c for c in changed.columns if c.startswith(("rms_", "kurtosis_", "crest_factor_", "fft_band_"))]:
        changed.loc[future, col] = changed.loc[future, col] * 1000.0
    second = build_health_indicator_samples(changed, fractions=fractions, healthy_window=20, recent_window=20)

    cols = health_indicator_feature_columns()
    np.testing.assert_allclose(first[cols].to_numpy(), second[cols].to_numpy(), rtol=0, atol=1e-12)
