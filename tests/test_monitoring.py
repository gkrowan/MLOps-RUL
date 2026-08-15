"""Tests for src/femto_rul/monitoring/ (Phase 17).

All of these run purely locally against synthetic DataFrames — no live
Postgres or network dependency, since Evidently's Report.run() is a local
computation.
"""

import json

import numpy as np
import pandas as pd
import pytest

from femto_rul.monitoring.column_mapping import build_data_definition
from femto_rul.monitoring.reference import load_reference_features
from femto_rul.monitoring.report import (
    build_report,
    drifted_column_share,
    prediction_sanity_summary,
    save_report,
)
from femto_rul.pipeline import FEATURE_COLUMNS_V1


def test_build_data_definition_matches_feature_columns_v1():
    data_definition = build_data_definition()
    assert data_definition.numerical_columns == FEATURE_COLUMNS_V1


def test_load_reference_features_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError):
        load_reference_features(missing)


def test_load_reference_features_missing_columns_raises(tmp_path):
    path = tmp_path / "train_features.parquet"
    pd.DataFrame({"rms_horiz": [1.0], "rul_seconds": [10.0]}).to_parquet(path)
    with pytest.raises(ValueError, match="missing expected Feature Set V1 columns"):
        load_reference_features(path)


def _synthetic_features(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(loc=shift, scale=1.0, size=n) for col in FEATURE_COLUMNS_V1}
    return pd.DataFrame(data)


def test_build_report_flags_an_obvious_shift(tmp_path):
    reference_df = _synthetic_features(200, seed=1)
    current_df = _synthetic_features(50, seed=2, shift=5.0)  # 5-sigma shift, every column

    snapshot = build_report(reference_df, current_df)
    share = drifted_column_share(snapshot)

    assert share is not None
    assert share > 0.5

    summary = prediction_sanity_summary(
        reference_targets=pd.Series([100.0, 200.0, 300.0]),
        current_predictions=pd.Series([150.0, 250.0]),
    )
    assert summary["training_rul_seconds"]["count"] == 3
    assert summary["production_predicted_rul_seconds"]["count"] == 2

    out_dir = save_report(snapshot, tmp_path / "run", summary)
    assert (out_dir / "data_drift.html").exists()
    assert (out_dir / "summary.json").exists()

    written = json.loads((out_dir / "summary.json").read_text())
    assert written["drifted_column_share"] == share
    assert written["prediction_sanity"] == summary


def test_build_report_no_drift_on_identical_distribution(tmp_path):
    reference_df = _synthetic_features(200, seed=1)
    current_df = _synthetic_features(50, seed=1)  # same seed, same distribution

    snapshot = build_report(reference_df, current_df)
    share = drifted_column_share(snapshot)

    assert share is not None
    assert share < 0.2
