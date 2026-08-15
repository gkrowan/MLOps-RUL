"""Best-effort operational logging for served predictions (Phase 16).

Every POST /predict call — success or failure — should produce exactly one
row in the `predictions` table, so Grafana's request-count/latency/error-rate
panels and Phase 17's Evidently drift reports have a real production window
to read instead of a mock.

A logging failure must never fail the prediction response. Callers (the
Phase 14 FastAPI endpoint) should wrap log_prediction in their own
try/except and log-and-continue on error, not propagate it to the client.
"""

from __future__ import annotations

from femto_rul import config
from femto_rul.db import get_connection
from femto_rul.pipeline import FEATURE_COLUMNS_V1

_STATUSES = ("ok", "error")


def log_prediction(
    *,
    request_id: str,
    model_name: str,
    model_version: str,
    model_alias: str,
    feature_set_version: str,
    features: dict[str, float] | None,
    predicted_rul_seconds: float | None,
    latency_ms: float,
    status: str,
    error_message: str | None = None,
) -> None:
    """Insert one row into the predictions table.

    features must contain exactly pipeline.FEATURE_COLUMNS_V1 on a
    successful ("ok") prediction; pass None (or an empty dict) on error.
    """
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {_STATUSES}, got {status!r}")

    features = features or {}
    if status == "ok":
        missing = set(FEATURE_COLUMNS_V1) - set(features)
        if missing:
            raise ValueError(
                f"missing feature columns for a successful prediction: {sorted(missing)}"
            )

    columns = [
        "request_id",
        "model_name",
        "model_version",
        "model_alias",
        "feature_set_version",
        *FEATURE_COLUMNS_V1,
        "predicted_rul_seconds",
        "latency_ms",
        "status",
        "error_message",
    ]
    values = [
        request_id,
        model_name,
        model_version,
        model_alias,
        feature_set_version,
        *[features.get(name) for name in FEATURE_COLUMNS_V1],
        predicted_rul_seconds,
        latency_ms,
        status,
        error_message,
    ]

    placeholders = ", ".join(["%s"] * len(values))
    column_list = ", ".join(columns)

    conn = get_connection(config.INFERENCE_DB_NAME)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {config.PREDICTIONS_TABLE} ({column_list}) "
                f"VALUES ({placeholders})",
                values,
            )
    finally:
        conn.close()
