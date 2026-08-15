"""Tests for src/femto_rul/serving/telemetry.py (Phase 16).

Validation tests run everywhere. The insert/round-trip tests need a real
`inference` Postgres database (docker compose up postgres) and skip cleanly
if one isn't reachable — per the project's rule against mocking database
behavior that production code actually depends on.
"""

import uuid

import pytest

from femto_rul import config
from femto_rul.db import get_connection
from femto_rul.pipeline import FEATURE_COLUMNS_V1
from femto_rul.serving.telemetry import log_prediction


def _require_db():
    try:
        conn = get_connection(config.INFERENCE_DB_NAME)
        conn.close()
    except Exception as exc:
        pytest.skip(f"inference Postgres not reachable: {exc}")


def test_log_prediction_rejects_invalid_status():
    with pytest.raises(ValueError, match="status must be"):
        log_prediction(
            request_id=str(uuid.uuid4()),
            model_name="m",
            model_version="1",
            model_alias="candidate",
            feature_set_version="v1",
            features={name: 0.0 for name in FEATURE_COLUMNS_V1},
            predicted_rul_seconds=1.0,
            latency_ms=1.0,
            status="pending",
        )


def test_log_prediction_rejects_missing_features_on_success():
    with pytest.raises(ValueError, match="missing feature columns"):
        log_prediction(
            request_id=str(uuid.uuid4()),
            model_name="m",
            model_version="1",
            model_alias="candidate",
            feature_set_version="v1",
            features={},
            predicted_rul_seconds=1.0,
            latency_ms=1.0,
            status="ok",
        )


def test_log_prediction_writes_a_row():
    _require_db()
    request_id = str(uuid.uuid4())
    log_prediction(
        request_id=request_id,
        model_name="femto-rul-model",
        model_version="1",
        model_alias="candidate",
        feature_set_version="v1",
        features={name: float(i) for i, name in enumerate(FEATURE_COLUMNS_V1)},
        predicted_rul_seconds=123.4,
        latency_ms=5.6,
        status="ok",
    )

    conn = get_connection(config.INFERENCE_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, predicted_rul_seconds, rms_horiz "
                "FROM predictions WHERE request_id = %s",
                (request_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    assert row == ("ok", 123.4, 0.0)


def test_log_prediction_writes_error_row_with_null_features():
    _require_db()
    request_id = str(uuid.uuid4())
    log_prediction(
        request_id=request_id,
        model_name="femto-rul-model",
        model_version="1",
        model_alias="candidate",
        feature_set_version="v1",
        features=None,
        predicted_rul_seconds=None,
        latency_ms=2.0,
        status="error",
        error_message="model timeout",
    )

    conn = get_connection(config.INFERENCE_DB_NAME)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, predicted_rul_seconds, rms_horiz, error_message "
                "FROM predictions WHERE request_id = %s",
                (request_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    assert row == ("error", None, None, "model timeout")
