import numpy as np
import pytest

from femto_rul.evaluation.metrics import phm12_score, regression_metrics


def test_regression_metrics_perfect_prediction():
    y = np.array([100.0, 50.0, 10.0])
    metrics = regression_metrics(y, y)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["phm12_snapshot_score"] == pytest.approx(1.0)


def test_phm12_penalizes_late_prediction_more_than_early_prediction():
    # Actual=100. Predicting 110 is -10% error (late / optimistic RUL).
    late = phm12_score([100.0], [110.0])
    # Predicting 90 is +10% error (early / conservative RUL).
    early = phm12_score([100.0], [90.0])
    assert late == pytest.approx(0.25)
    assert early == pytest.approx(np.sqrt(0.5))
    assert late < early


def test_phm12_skips_zero_rul_rows():
    score = phm12_score([100.0, 0.0], [100.0, 100.0])
    assert score == pytest.approx(1.0)
