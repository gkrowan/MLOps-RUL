CREATE DATABASE airflow;
CREATE DATABASE mlflow;
CREATE DATABASE inference;

-- Phase 16 — prediction logging. Lives in its own database (not the mlflow
-- backend store) so operational telemetry has no coupling to MLflow's
-- internal schema. See docs/phase_16_prediction_logging_and_grafana.md.
\c inference

CREATE TABLE IF NOT EXISTS predictions (
    id                    BIGSERIAL PRIMARY KEY,
    request_id            UUID NOT NULL UNIQUE,
    requested_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    model_name            TEXT NOT NULL,
    model_version         TEXT NOT NULL,
    model_alias           TEXT NOT NULL,
    feature_set_version   TEXT NOT NULL,

    -- Prefix V1 — the actual feature set the registered model consumes
    -- (src/femto_rul/features/prefix.py:prefix_feature_columns() is the
    -- source of truth for these names and their order).
    observed_age_seconds                       DOUBLE PRECISION,
    rotation_speed_rpm                         DOUBLE PRECISION,
    radial_load_n                              DOUBLE PRECISION,
    rms_horiz_current_over_early               DOUBLE PRECISION,
    rms_horiz_recent_mean_over_early           DOUBLE PRECISION,
    rms_horiz_recent_slope_per_hour            DOUBLE PRECISION,
    rms_vert_current_over_early                DOUBLE PRECISION,
    rms_vert_recent_mean_over_early            DOUBLE PRECISION,
    rms_vert_recent_slope_per_hour             DOUBLE PRECISION,
    kurtosis_horiz_current_over_early          DOUBLE PRECISION,
    kurtosis_horiz_recent_mean_over_early      DOUBLE PRECISION,
    kurtosis_horiz_recent_slope_per_hour       DOUBLE PRECISION,
    kurtosis_vert_current_over_early           DOUBLE PRECISION,
    kurtosis_vert_recent_mean_over_early       DOUBLE PRECISION,
    kurtosis_vert_recent_slope_per_hour        DOUBLE PRECISION,
    crest_factor_horiz_current_over_early      DOUBLE PRECISION,
    crest_factor_horiz_recent_mean_over_early  DOUBLE PRECISION,
    crest_factor_horiz_recent_slope_per_hour   DOUBLE PRECISION,
    crest_factor_vert_current_over_early       DOUBLE PRECISION,
    crest_factor_vert_recent_mean_over_early   DOUBLE PRECISION,
    crest_factor_vert_recent_slope_per_hour    DOUBLE PRECISION,

    predicted_rul_seconds DOUBLE PRECISION,
    latency_ms            DOUBLE PRECISION NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error_message           TEXT
);

CREATE INDEX IF NOT EXISTS ix_predictions_requested_at ON predictions (requested_at);
CREATE INDEX IF NOT EXISTS ix_predictions_status ON predictions (status);
