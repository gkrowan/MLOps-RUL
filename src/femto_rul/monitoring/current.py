"""Loads the current (production) window from the predictions table
(Phase 17), written by src/femto_rul/serving/telemetry.py (Phase 16)."""

from __future__ import annotations

import pandas as pd

from femto_rul import config
from femto_rul.db import get_connection
from femto_rul.features.prefix import prefix_feature_columns


def load_current_features(window: str = "24 hours") -> pd.DataFrame:
    """Feature vectors for successful predictions in the trailing `window`
    (a Postgres interval literal, e.g. "24 hours", "7 days").

    status='error' rows are excluded — their feature columns are NULL by
    construction (a failed prediction never populated them), and including
    them would just show up as 100% missing-value drift, a signal already
    visible directly in Grafana's error-rate panel.
    """
    columns = ", ".join(prefix_feature_columns())
    query = f"""
        SELECT {columns}
        FROM {config.PREDICTIONS_TABLE}
        WHERE status = 'ok'
          AND requested_at >= now() - %(window)s::interval
    """

    conn = get_connection(config.INFERENCE_DB_NAME)
    try:
        return pd.read_sql(query, conn, params={"window": window})
    finally:
        conn.close()


def load_current_predictions(window: str = "24 hours") -> pd.Series:
    """predicted_rul_seconds for successful predictions in the trailing
    window — used for the prediction-distribution sanity check."""
    query = f"""
        SELECT predicted_rul_seconds
        FROM {config.PREDICTIONS_TABLE}
        WHERE status = 'ok'
          AND requested_at >= now() - %(window)s::interval
    """

    conn = get_connection(config.INFERENCE_DB_NAME)
    try:
        return pd.read_sql(query, conn, params={"window": window})["predicted_rul_seconds"]
    finally:
        conn.close()
