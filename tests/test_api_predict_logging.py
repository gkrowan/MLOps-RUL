"""Tests that POST /predict logs to the predictions table (Phase 16 Change 4).

No MLflow model is registered in this dev environment, so `api.main.app`'s
model-loading lifespan is never run here — TestClient(app) used without the
`with` block skips lifespan entirely. Instead `model_state["model"]` is
monkeypatched to a trivial stub, keeping the test focused on the thing it's
actually verifying: that a real /predict call writes a real row to the real
`predictions` table. Skips cleanly if Postgres isn't reachable, same as
tests/test_telemetry.py.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main
from femto_rul import config
from femto_rul.db import get_connection

client = TestClient(main.app)

_VALID_PAYLOAD = {name: 1.0 for name in main.EXPECTED_FEATURES}


def _require_db():
    try:
        conn = get_connection(config.INFERENCE_DB_NAME)
        conn.close()
    except Exception as exc:
        pytest.skip(f"inference Postgres not reachable: {exc}")


class _StubModel:
    def __init__(self, value: float | None = None, error: Exception | None = None):
        self._value = value
        self._error = error

    def predict(self, df):
        if self._error is not None:
            raise self._error
        return np.array([self._value])


def test_predict_success_logs_an_ok_row(monkeypatch):
    _require_db()
    monkeypatch.setitem(main.model_state, "model", _StubModel(value=123.4))
    monkeypatch.setitem(main.model_state, "version", "1")
    monkeypatch.setitem(main.model_state, "run_id", "run-1")

    response = client.post("/predict", json=_VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_rul_seconds"] == pytest.approx(123.4)

    # log_prediction generates its own request_id server-side; find the row
    # by looking it up as the most recent successful insert instead.
    conn = get_connection(config.INFERENCE_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, predicted_rul_seconds, feature_set_version, "
                "observed_age_seconds, error_message "
                "FROM predictions ORDER BY requested_at DESC LIMIT 1"
            )
            row = cur.fetchone()
    finally:
        conn.close()

    assert row == ("ok", pytest.approx(123.4), "prefix_v1", 1.0, None)


def test_predict_failure_logs_an_error_row(monkeypatch):
    _require_db()
    monkeypatch.setitem(
        main.model_state, "model", _StubModel(error=RuntimeError("model exploded"))
    )
    monkeypatch.setitem(main.model_state, "version", "1")
    monkeypatch.setitem(main.model_state, "run_id", "run-1")

    response = client.post("/predict", json=_VALID_PAYLOAD)
    assert response.status_code == 500

    conn = get_connection(config.INFERENCE_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, predicted_rul_seconds, observed_age_seconds, error_message "
                "FROM predictions ORDER BY requested_at DESC LIMIT 1"
            )
            row = cur.fetchone()
    finally:
        conn.close()

    assert row == ("error", None, None, "model exploded")


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setitem(main.model_state, "model", None)

    response = client.post("/predict", json=_VALID_PAYLOAD)
    assert response.status_code == 503
