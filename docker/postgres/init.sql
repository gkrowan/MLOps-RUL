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

    -- Feature Set V1 — same names/units as train_features.parquet
    -- (src/femto_rul/pipeline.py:FEATURE_COLUMNS_V1 is the source of truth).
    rms_horiz             DOUBLE PRECISION,
    kurtosis_horiz        DOUBLE PRECISION,
    skewness_horiz        DOUBLE PRECISION,
    crest_factor_horiz    DOUBLE PRECISION,
    fft_band_0_horiz      DOUBLE PRECISION,
    fft_band_1_horiz      DOUBLE PRECISION,
    fft_band_2_horiz      DOUBLE PRECISION,
    fft_band_3_horiz      DOUBLE PRECISION,
    fft_band_4_horiz      DOUBLE PRECISION,
    fft_band_5_horiz      DOUBLE PRECISION,
    fft_band_6_horiz      DOUBLE PRECISION,
    fft_band_7_horiz      DOUBLE PRECISION,
    rms_vert              DOUBLE PRECISION,
    kurtosis_vert         DOUBLE PRECISION,
    skewness_vert         DOUBLE PRECISION,
    crest_factor_vert     DOUBLE PRECISION,
    fft_band_0_vert       DOUBLE PRECISION,
    fft_band_1_vert       DOUBLE PRECISION,
    fft_band_2_vert       DOUBLE PRECISION,
    fft_band_3_vert       DOUBLE PRECISION,
    fft_band_4_vert       DOUBLE PRECISION,
    fft_band_5_vert       DOUBLE PRECISION,
    fft_band_6_vert       DOUBLE PRECISION,
    fft_band_7_vert       DOUBLE PRECISION,

    predicted_rul_seconds DOUBLE PRECISION,
    latency_ms            DOUBLE PRECISION NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error_message           TEXT
);

CREATE INDEX IF NOT EXISTS ix_predictions_requested_at ON predictions (requested_at);
CREATE INDEX IF NOT EXISTS ix_predictions_status ON predictions (status);
