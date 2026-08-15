# Phase 16 — Prediction Logging + Grafana

**Branch:** `feature/e2e-mlops-pipeline`
**Parent:** latest `main`
**Purpose:** Give every inference request a durable, queryable record, and expose that record as an operational Grafana dashboard.

---

## Scaffolding status (2026-08-14)

Built ahead of Phase 14/15, since none of it requires a live API to exist yet:

```text
DONE  docker/postgres/init.sql        — inference DB + predictions table + indexes
DONE  src/femto_rul/config.py         — INFERENCE_DB_HOST/PORT/NAME, PREDICTIONS_TABLE, POSTGRES_USER
DONE  src/femto_rul/pipeline.py       — FEATURE_COLUMNS_V1 (single source of truth for column order)
DONE  src/femto_rul/db.py             — shared Postgres connection helper
DONE  src/femto_rul/serving/telemetry.py — log_prediction(), tested
DONE  grafana/provisioning/datasources/inference-postgres.yml
DONE  grafana/provisioning/dashboards/{dashboards.yml, prediction_logging.json}
DONE  requirements.txt                — psycopg2-binary==2.9.10
DONE  tests/test_telemetry.py         — validation tests run always; insert/round-trip
                                         tests skip cleanly without a reachable Postgres

STILL BLOCKED ON PHASE 14  — nothing calls log_prediction() yet; there is no
                              /predict endpoint. §5 Change 4 below describes
                              the integration point once api/ exists.
```

One deliberate deviation from the original plan below: `telemetry.py` lives in
`src/femto_rul/serving/`, not `api/`. It has no FastAPI dependency — it's a
plain function that takes primitives and writes a row — so it belongs in the
installable package where `tests/` can import it directly, rather than
presupposing Phase 14's `api/` layout. Phase 14's endpoint will simply
`from femto_rul.serving.telemetry import log_prediction` and call it.

---

# 1. Phase 16 Outcome

At the end of Phase 16 we want:

```text
FastAPI /predict request
        ↓
prediction + latency + status captured
        ↓
row written to a dedicated inference database
        ↓
Grafana panel reads that table
        ↓
request count / latency / error rate / active model version visible live
```

Phase 16 does **not** implement drift detection — that is Phase 17.

Phase 16 does **not** implement the drift simulation harness — that is Phase 18.

Phase 16 does **not** build FastAPI or Docker — those are Phases 14–15 and must exist first (see §2).

It gives Phases 17 and 18 a real table to read from instead of a mock.

---

# 2. Dependency Gate

Phase 16 cannot execute against a live endpoint until these exist:

```text
Phase 14 — api/main.py, api/schemas.py, api/model_loader.py, POST /predict
Phase 15 — Dockerfile serving that API
```

Neither exists in the repository yet (`find api -type f` returns nothing as of this writing). This document is written ahead of that work so the logging contract is fixed before `/predict` is coded, per the project plan's guidance that C (deployment) and D (monitoring) shouldn't sit idle waiting on a "final" model.

What **does** already exist and Phase 16 reuses rather than rebuilds:

```text
docker-compose.yml   → postgres, mlflow, grafana services already defined
grafana/provisioning/datasources/postgres.yml → a Postgres datasource already provisioned
docs/e2e_mlops_pipeline_phasewise_rollout.md   → §4 confirms Grafana is Chris's owned infra
```

Phase 16 extends this infrastructure; it does not stand up new services.

---

# 3. Known Issue to Fix

`grafana/provisioning/datasources/postgres.yml` currently points at:

```yaml
jsonData:
  database: mlflow
```

That is the MLflow backend store database — it holds MLflow's own run/experiment metadata tables, not application telemetry. Querying it directly for request counts or latency would mean parsing MLflow's internal schema, which is not a contract we control or want to depend on.

Decision:

```text
Do not repoint or mutate the existing "MLOps PostgreSQL" datasource.
Chris/B may already depend on it for something else.
Add a second, purpose-built datasource instead.
```

Phase 16 adds its own database (`inference`) and its own Grafana datasource (`inference-postgres.yml`), left additive so nothing already provisioned breaks.

---

# 4. Prediction Log Schema

## Design decision: one wide table, not JSONB

Feature Set V1 is fixed at 24 columns (12 horizontal + 12 vertical, per `docs/phase_01_foundation_and_reuse.md` §Change 12). That's small enough to store as typed columns rather than a JSONB blob. Doing so means:

- Grafana can query latency/error/count panels directly with SQL, no JSON path expressions
- Phase 17 (Evidently) can `pandas.read_sql` the same table as its "current" production distribution — no second storage path to keep in sync with `train_features.parquet`'s column names

If the feature set grows unbounded later (v2+), revisit this; for v1 scope a wide table is simpler than premature normalization.

## Target table: `predictions`

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id                    BIGSERIAL PRIMARY KEY,
    request_id            UUID NOT NULL UNIQUE,
    requested_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    model_name            TEXT NOT NULL,
    model_version         TEXT NOT NULL,
    model_alias           TEXT NOT NULL,
    feature_set_version   TEXT NOT NULL,

    -- Feature Set V1 — same names/units as train_features.parquet
    rms_horiz             FLOAT8,
    kurtosis_horiz        FLOAT8,
    skewness_horiz        FLOAT8,
    crest_factor_horiz    FLOAT8,
    fft_band_0_horiz      FLOAT8,
    fft_band_1_horiz      FLOAT8,
    fft_band_2_horiz      FLOAT8,
    fft_band_3_horiz      FLOAT8,
    fft_band_4_horiz      FLOAT8,
    fft_band_5_horiz      FLOAT8,
    fft_band_6_horiz      FLOAT8,
    fft_band_7_horiz      FLOAT8,
    rms_vert              FLOAT8,
    kurtosis_vert         FLOAT8,
    skewness_vert         FLOAT8,
    crest_factor_vert     FLOAT8,
    fft_band_0_vert       FLOAT8,
    fft_band_1_vert       FLOAT8,
    fft_band_2_vert       FLOAT8,
    fft_band_3_vert       FLOAT8,
    fft_band_4_vert       FLOAT8,
    fft_band_5_vert       FLOAT8,
    fft_band_6_vert       FLOAT8,
    fft_band_7_vert       FLOAT8,

    predicted_rul_seconds FLOAT8,
    latency_ms            FLOAT8 NOT NULL,
    status                TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error_message          TEXT
);

CREATE INDEX IF NOT EXISTS ix_predictions_requested_at ON predictions (requested_at);
CREATE INDEX IF NOT EXISTS ix_predictions_status ON predictions (status);
```

Notes:

```text
request_id     — generated server-side (uuid4), also returned in the API response
                 so a caller can correlate a prediction with a log row
model_version  — resolved registry version number (e.g. "3"), not the alias,
                 so a Grafana panel can show exactly which artifact answered
model_alias    — "candidate" / "champion" (config.MLFLOW_MODEL_ALIAS)
status='error' — feature columns and predicted_rul_seconds are NULL;
                 error_message holds the failure reason
```

Rows where `status='error'` must still be written. An endpoint that only logs successes cannot report an accurate error rate.

---

# 5. Exact Repository Changes

## Change 1 — Add the `inference` database (DONE)

`docker/postgres/init.sql` currently creates `airflow` and `mlflow`. Add:

```sql
CREATE DATABASE airflow;
CREATE DATABASE mlflow;
CREATE DATABASE inference;
```

`init.sql` only runs on first container init (fresh volume). Document in the phase 16 PR that anyone with an existing `postgres-data` volume needs to either recreate it or run `CREATE DATABASE inference;` manually — do not silently assume a clean environment.

---

## Change 2 — Config additions (DONE)

`src/femto_rul/config.py` already has `POSTGRES`-adjacent values coming through `.env` (via Docker) but no Python constants for the inference DB. Add, following the existing `env()` helper pattern:

```python
# ---------------------------------------------------------------------------
# Inference telemetry (Phase 16)
# ---------------------------------------------------------------------------

INFERENCE_DB_NAME: Final[str] = (
    env("INFERENCE_DB_NAME", "inference") or "inference"
)

PREDICTIONS_TABLE: Final[str] = "predictions"
```

Do not add a full `INFERENCE_DATABASE_URL` env var — per the Phase 1 config policy, stable service composition (host `postgres`, port `5432`, user/password already in `POSTGRES_USER`/`POSTGRES_PASSWORD`) should be assembled in code from existing env vars, not duplicated as a new secret-bearing string to keep in sync.

---

## Change 3 — Networking: FastAPI must join the compose network (still applies to Phase 15)

`postgres` has no `ports:` mapping in `docker-compose.yml` — only `mlflow` and `airflow` (both containers on the same compose network) can currently reach `postgres:5432`. A FastAPI process run bare on a laptop cannot.

Decision for Phase 15/16 integration:

```text
The FastAPI service (Phase 15 Docker image) must run as a service
in docker-compose.yml, on the same network, so it resolves:
    postgres:5432
    mlflow:5000
the same way the airflow containers already do.
```

This is consistent with Phase 15's own acceptance gate ("reaches MLflow"). Flag this now so Phase 15's Dockerfile/compose entry isn't built assuming `localhost` service URLs that only work from a host shell.

---

## Change 4 — Logging module (DONE, built in `src/femto_rul/serving/`)

Built as `src/femto_rul/db.py` (shared connection helper, host/port from
`config.INFERENCE_DB_HOST`/`INFERENCE_DB_PORT`, password via
`config.require_env("POSTGRES_PASSWORD")` at connection time) and
`src/femto_rul/serving/telemetry.py` (`log_prediction(...)`), not `api/` —
see the deviation note at the top of this document for why. Validates
`status` and, on `status="ok"`, that every `pipeline.FEATURE_COLUMNS_V1`
column is present before opening a connection, so a malformed call fails
fast with a clear `ValueError` instead of a partial insert.

Phase 14's `POST /predict` will call it like this once it exists:

```python
import time
import uuid

from femto_rul.serving.telemetry import log_prediction

request_id = str(uuid.uuid4())
start = time.perf_counter()
try:
    prediction = model.predict(features)
    status, error_message = "ok", None
except Exception as exc:
    prediction, status, error_message = None, "error", str(exc)
latency_ms = (time.perf_counter() - start) * 1000

try:
    log_prediction(
        request_id=request_id, model_name=..., model_version=..., model_alias=...,
        feature_set_version="v1", features=features if status == "ok" else None,
        predicted_rul_seconds=prediction, latency_ms=latency_ms,
        status=status, error_message=error_message,
    )
except Exception:
    logger.exception("prediction logging failed")  # best-effort; never fail the response
```

A logging failure must never fail the prediction response — the `try/except`
around `log_prediction` at the call site is Phase 14's responsibility, not
`telemetry.py`'s; the module itself raises on bad input so callers *know*
when they've built a malformed call during development.

---

## Change 5 — Grafana datasource (additive) (DONE)

New file: `grafana/provisioning/datasources/inference-postgres.yml`

```yaml
apiVersion: 1

datasources:
  - name: Inference PostgreSQL
    uid: inference-postgres
    type: postgres
    access: proxy
    url: postgres:5432
    user: ${POSTGRES_USER}
    secureJsonData:
      password: ${POSTGRES_PASSWORD}
    jsonData:
      database: inference
      sslmode: disable
      postgresVersion: 1600
    isDefault: false
```

`isDefault: false` so it doesn't compete with the existing "MLOps PostgreSQL" datasource. `grafana/provisioning/` is already bind-mounted whole in `docker-compose.yml`, so no compose change is needed — dropping the file in is enough.

---

## Change 6 — Grafana dashboard provisioning (DONE)

No `grafana/provisioning/dashboards/` directory exists yet — only `datasources/`. Add:

`grafana/provisioning/dashboards/dashboards.yml` (provider config):

```yaml
apiVersion: 1

providers:
  - name: femto-rul
    folder: FEMTO RUL
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

`grafana/provisioning/dashboards/prediction_logging.json` — dashboard with panels:

```text
Request count      — count(*) from predictions, bucketed by $__interval
Latency (p50/p95)  — percentile_cont(latency_ms) over time
Error rate         — count(status='error') / count(*), rolling window
Throughput         — requests per minute
Active model       — Stat panel: most recent (model_name, model_version, model_alias)
```

Each panel's query targets the `inference-postgres` datasource uid, not the default one.

---

## Change 7 — Tests (DONE — validation tests; insert tests skip without Postgres)

`api/` doesn't exist yet, so Phase 16 tests can only be written once Phase 14 lands. Plan for:

```text
tests/test_telemetry.py
```

covering:

- `log_prediction` inserts exactly one row with all 24 feature columns populated on success
- `log_prediction` inserts a row with `status='error'`, null feature/prediction columns, and a non-null `error_message` on failure
- inserted `request_id` round-trips to what the caller generated

These need a real Postgres to run against (or `testcontainers`) — do not mock the insert, per the "don't mock what causes production divergence" lesson that applies broadly to this kind of integration surface. Mark them to skip cleanly in CI if no Postgres is reachable, rather than failing the whole suite.

---

# 6. Proposed File Tree (after Phase 16)

```text
MLOps-RUL/
├── docker/postgres/init.sql                       (+ inference database + predictions table) DONE
├── grafana/provisioning/
│   ├── datasources/
│   │   ├── postgres.yml                            (unchanged)
│   │   └── inference-postgres.yml                  DONE
│   └── dashboards/
│       ├── dashboards.yml                          DONE
│       └── prediction_logging.json                 DONE
├── src/femto_rul/
│   ├── config.py                                   (+ inference/monitoring constants) DONE
│   ├── pipeline.py                                  (+ FEATURE_COLUMNS_V1) DONE
│   ├── db.py                                        DONE
│   └── serving/
│       ├── __init__.py                              DONE
│       └── telemetry.py                             DONE
├── tests/test_telemetry.py                          DONE
└── docs/phase_16_prediction_logging_and_grafana.md  (this file)
```

Still to come, blocked on Phase 14/15: `api/` itself, and actually calling `log_prediction()` from a live endpoint.

---

# 7. Phase 16 Acceptance Criteria

- [x] `inference` database exists and is created by `init.sql` on a fresh volume
- [x] `predictions` table matches the schema in §4
- [ ] every `/predict` call — success or failure — produces exactly one row (blocked: no `/predict` yet)
- [x] a logging failure does not fail the prediction response *(by contract — `telemetry.py` raises on bad input, doesn't swallow errors; the call site in Phase 14 is responsible for catch-and-continue)*
- [x] `inference-postgres.yml` datasource is added without modifying the existing `postgres.yml`
- [x] Grafana dashboard shows request count, latency, error rate, throughput, active model version
- [ ] FastAPI service reaches `postgres:5432` via the compose network, not `localhost` (blocked on Phase 15)
- [x] no credentials are hard-coded in `telemetry.py`/`db.py` (reused from existing env vars)
- [x] tests exist and skip cleanly without a reachable Postgres (verified: `pytest -q` → 2 skipped, rest pass)

---

# 8. What Phase 16 Explicitly Does NOT Do

```text
drift detection            → Phase 17
drift simulation            → Phase 18
alerting rules / thresholds → out of scope for the course project unless time remains
authentication on /predict  → out of scope
retry/queue-based logging   → best-effort synchronous insert is sufficient at this scale
```

---

# 9. Suggested Commit Sequence

Matches item 19 in the e2e doc's commit sequence: `feat: add inference telemetry`.

```text
1. chore: add inference database and predictions table
2. feat: add prediction logging module (api/telemetry.py)
3. feat: provision Grafana inference datasource and dashboard
4. test: add prediction logging tests
```

---

# 10. Next Phase

```text
Phase 17
=
Evidently reference distribution (train_features.parquet)
+
Evidently current distribution (predictions table, this phase's output)
+
data drift / data quality reports
```

Phase 17 reads directly from the `predictions` table this phase creates — see `docs/phase_17_evidently_monitoring.md`.
