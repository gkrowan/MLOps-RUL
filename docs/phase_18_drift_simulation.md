# Phase 18 — Drift Simulation

# 1. Phase 18 Outcome

At the end of Phase 18 we want:

```text
engineered "drifted" feature payloads
        ↓
POSTed through the live /predict endpoint (real HTTP, not in-process)
        ↓
logged into the predictions table by Phase 16's existing pipeline
        ↓
Phase 17's reference-vs-current pipeline re-run, scoped to just-injected traffic
        ↓
HTML report + evidence saved under artifacts/monitoring/drift_simulation/,
proving Phase 17's reports actually react to abnormal input
```

Phase 18 does **not** build new monitoring logic — it is a consumer of Phase 17's existing `column_mapping.py` / `reference.py` / `current.py` / `report.py`, reused as-is. Its only new code is (a) functions that construct abnormal-but-realistic request payloads, and (b) a CLI script that sends them through the real API and runs the existing report.

Phase 18 does **not** evaluate whether the *model* is right about anything — same as Phase 17, it has no ground truth for live traffic. It only proves the *monitoring* reacts to distributional/schema abnormality.

---

# 2. Dependency Gate

```text
live /predict endpoint → api/main.py                    (Phase 14/15 output — already built and working)
prediction logging     → predictions table               (Phase 16 output — already built and working)
reference distribution → data/processed/prefix_train_v1.parquet (Phase 6 output — built 2026-08-15)
reference bounds       → configs/monitoring_reference_ranges.json (Phase 17 output — generated 2026-08-15)
reporting pipeline     → src/femto_rul/monitoring/*.py   (Phase 17 output — already built and working)
```

Everything Phase 18 depends on already exists and has been exercised end-to-end (2026-08-15 generated the first real Evidently report from live traffic). Phase 18 is unblocked and can be built immediately — unlike Phase 17, which had to wait on Phase 16.

## Key finding that shapes this doc — verify before implementing

The live `BearingFeatures` request model (`api/main.py:171-178`, built via `create_model()` from `prefix_feature_columns()`) has **zero numeric range constraints** — every field is `(float, ...)`, so any finite float passes Pydantic and reaches `log_prediction()`. Only a genuinely malformed request (missing/extra/misspelled field, non-numeric value) triggers Pydantic's automatic HTTP 422 — and that rejection happens *before* the endpoint body runs, so it is **never** logged to `predictions` and Evidently **never** sees it.

Practical consequence: the rollout doc's phrase "missing values, invalid schema, extreme values" (one nominal scenario) actually splits into two different mechanisms that must be evidenced differently:

```text
missing-value sentinel (e.g. -999.0) → schema-valid, passes Pydantic → logged 'ok' → Evidently sees it
extreme values (e.g. 10x reference p99) → schema-valid, passes Pydantic → logged 'ok' → Evidently sees it
invalid schema (missing/extra/wrong-type field) → 422, rejected before logging → Evidently NEVER sees it
```

This doc treats the first two as Evidently-report evidence, and the third as a separate boundary-check artifact (the API's own input validation, not the monitoring pipeline) — see §4.

---

# 3. Scenario Definitions

## Scenario 1 — Amplitude drift

Simulates rising vibration energy (e.g. bearing degradation). Scales the ratio-type columns (`*_current_over_early`, `*_recent_mean_over_early`, 12 of the 21 columns) on a real sampled row so the result lands well beyond that column's `p99` from `configs/monitoring_reference_ranges.json`.

## Scenario 2 — Channel drift

Simulates a sensor/wiring swap. Swaps every `_horiz`/`_vert` column pair (9 pairs, all 18 derived feature columns) on a real sampled row. The 3 context columns (`observed_age_seconds`, `rotation_speed_rpm`, `radial_load_n`) are untouched — they aren't per-channel.

## Scenario 3 — Data-quality anomaly (two mechanisms, see §2)

**3a. Missing-value sentinel** — a subset of columns overwritten with a sentinel (e.g. `-999.0`). Schema-valid, logged, visible to Evidently as an extreme/out-of-range value (a live model has no way to receive a true `null` for a required field — the sentinel is the realistic stand-in).

**3b. Extreme values** — a subset of columns pushed to a large multiple beyond `p01`/`p99`. Schema-valid, logged, visible to Evidently.

**3c. Invalid schema** — deliberately malformed raw request bodies (missing a required field; an extra/misspelled field, since the model is `extra="forbid"`; a string where a float is expected). Sent as literal HTTP requests, not run through any local validation first. Expected outcome: HTTP 422, zero `predictions` rows, zero Evidently visibility. Evidence is the saved request/response pair itself, proving the API's own input boundary is the defense for this case — explicitly labeled as a different layer than monitoring, not folded into the drift report.

All perturbations start from real rows sampled from `load_reference_features()` (`src/femto_rul/monitoring/reference.py`, reused as-is), not fabricated from scratch, so payloads look like real bearings apart from the injected anomaly.

---

# 4. Evidence & Verification Strategy

For scenarios 1, 2, 3a, 3b: after POSTing a batch, immediately pull `load_current_features(window="2 minutes")` (`src/femto_rul/monitoring/current.py`, reused as-is — a tight window keeps this scenario's rows isolated from unrelated traffic), then reuse `build_report()` / `drifted_column_share()` / `save_report()` (`src/femto_rul/monitoring/report.py`, the exact same functions `tests/test_monitoring.py` already exercises against synthetic drift) to produce `artifacts/monitoring/drift_simulation/<scenario>/<timestamp>/`.

For scenario 3c: no Evidently call at all. Save `artifacts/monitoring/drift_simulation/schema_boundary/<timestamp>/evidence.json`, one entry per malformed request: `{payload, status_code, response_body}`.

This directly satisfies the rollout doc's stated intent (`docs/e2e_mlops_pipeline_phasewise_rollout.md`, "save monitoring evidence for the presentation") and Phase 17's own forward-reference (`docs/phase_17_evidently_monitoring.md`, "verify this phase's reports react").

---

# 5. Exact Repository Changes

## Change 1 — Dependency (DONE)

`requirements.txt` needs a new `requests==<pinned>` block (with an explanatory comment, matching the style of the `fastapi==...` addition in Phase 16) — nothing in the repo currently makes real outbound HTTP calls; the existing test suite only uses FastAPI's in-process `TestClient`.

## Change 2 — Scenario module (DONE)

New: `src/femto_rul/monitoring/drift_scenarios.py`. Pure, DB/network-free functions, same testability convention as `column_mapping.py`/`report.py`:

```python
def amplitude_drift(row: dict, reference_ranges: dict, factor: float = 4.0) -> dict: ...
def channel_drift(row: dict) -> dict: ...
def missing_value_sentinel(row: dict, columns: list[str], sentinel: float = -999.0) -> dict: ...
def extreme_values(row: dict, reference_ranges: dict, columns: list[str], multiplier: float = 10.0) -> dict: ...
def invalid_schema_requests() -> list[dict]: ...
```

Reuses `prefix_feature_columns()` (`src/femto_rul/features/prefix.py`) for column names/grouping. No Evidently or Postgres logic lives here.

## Change 3 — Config addition (DONE)

`src/femto_rul/config.py` gets `API_BASE_URL`, added alongside the existing `AIRFLOW_BASE_URL`/`MLFLOW_TRACKING_URI`/`GRAFANA_BASE_URL` constants in the "External MLOps services" block (`config.py:173-182`), not a new section — it's the same kind of value.

```python
API_BASE_URL: Final[str | None] = env("API_BASE_URL")
```

## Change 4 — CLI entry point (DONE, live-verified 2026-08-17)

New: `scripts/simulate_drift.py`. Same convention as `run_monitoring_report.py`/`generate_reference_ranges.py`: bare `argparse.ArgumentParser()`, `--kebab-case` flags, `print()` progress, `main()` guarded by `__main__`.

```text
Usage: python scripts/simulate_drift.py [--scenario amplitude|channel|quality|all]
                                         [--api-url http://localhost:8000]
                                         [--n-requests 15]
                                         [--out artifacts/monitoring/drift_simulation]
```

Per scenario: sample `--n-requests` base rows → perturb via `drift_scenarios` → POST to `{api_url}/predict` via `requests` → run the evidence strategy from §4 → print the drifted-column share (or, for 3c, the count of 422s).

## Change 5 — Tests (DONE, 5 passed)

New: `tests/test_drift_scenarios.py`, mirroring `tests/test_monitoring.py`'s synthetic-data-only style — no network/DB dependency:

- `amplitude_drift`/`extreme_values` push values past the given bounds by the expected factor.
- `channel_drift` swaps exactly the 18 `_horiz`/`_vert` columns and leaves the 3 context columns untouched.
- `missing_value_sentinel` overwrites exactly the requested columns.
- `invalid_schema_requests()` returns payloads that are actually malformed relative to `prefix_feature_columns()`.

---

# 6. Proposed File Tree (after Phase 18)

```text
MLOps-RUL/
├── src/femto_rul/
│   ├── config.py                                  (+ API_BASE_URL) DONE
│   └── monitoring/
│       └── drift_scenarios.py                     DONE
├── scripts/
│   └── simulate_drift.py                          DONE, live-verified 2026-08-17
├── artifacts/monitoring/drift_simulation/          generated 2026-08-17, gitignored
│   ├── amplitude/<timestamp>/
│   ├── channel/<timestamp>/
│   ├── quality/<timestamp>/
│   └── schema_boundary/<timestamp>/evidence.json
├── tests/test_drift_scenarios.py                   DONE, 5 passed
└── docs/phase_18_drift_simulation.md               (this file)
```

---

# 7. Phase 18 Acceptance Criteria

- [x] `requests` pinned in `requirements.txt`
- [x] `drift_scenarios.py` functions are pure and unit-tested without network/DB dependency
- [x] amplitude/channel/quality-extreme scenarios POST through the real `/predict` HTTP endpoint (not an in-process bypass) and get logged to `predictions` — verified live 2026-08-17, `femto-rul-model@champion` (E101 median baseline) registered via the real `register_finalists.py` → `evaluate_official_holdout.py` → `promote_champion.py` pipeline
- [x] each of those scenarios re-runs Phase 17's existing `build_report`/`save_report` scoped to a tight time window and shows an elevated drifted-column share vs. the ~0.0–0.2 no-drift baseline already established in `tests/test_monitoring.py` — amplitude 0.667, channel 0.238, quality 0.286 (all above baseline; channel is the weakest signal, see note below)
- [x] invalid-schema requests are confirmed to 422 and produce zero `predictions` rows, with the request/response evidence saved separately from the Evidently reports — verified: all 3 payloads returned 422, zero rows landed in `predictions` in that window
- [x] `pytest tests/test_drift_scenarios.py -q` passes with no infra running (5 passed)

**Live run 2026-08-17**: `python scripts/simulate_drift.py --scenario all --n-requests 15` against a locally-registered champion model. Results: amplitude drifted_column_share=0.667 (16 rows), channel=0.238 (15 rows), quality=0.286 (30 rows, missing-value-sentinel + extreme-value combined), schema_boundary: 3/3 payloads 422'd, 0 rows logged. Evidence under `artifacts/monitoring/drift_simulation/` (gitignored, local only).

**Note on channel drift's weaker signal**: 0.238 clears the ~0.2 no-drift baseline but far less decisively than amplitude's 0.667. Evidently's `DataDriftPreset` tests each column's *marginal* distribution independently — swapping horiz/vert values only shows up as drift where those two channels' marginal distributions actually differ; where they're similar, a swap is statistically invisible to a per-column test even though it's a real data-quality problem. This is an inherent limitation of marginal drift detection, not a bug in the scenario — worth calling out if this comes up in the presentation.

**Two real bugs found and fixed while getting live verification working (unrelated to drift-simulation logic itself)**:
1. `docker/api/requirements.txt` was missing `scikit-learn` — `register_finalists.py` logs every finalist via the `mlflow.sklearn` flavor regardless of algorithm, so the API container couldn't load *any* registered model, not just this session's. Fixed by adding the same `scikit-learn==1.9.0` pin used elsewhere in the repo.
2. A `.env` file appeared in the repo root (created 2026-08-17, not by this session) holding real shared-VM credentials, explicitly labeled in its own header as "the existing cloud environment." Docker Compose auto-loads `.env`, so any `docker compose` invocation without explicit overrides silently picked up those cloud credentials against the local-only containers, breaking Postgres auth. Confirmed with the user this file should be ignored for local work; all local `docker compose` commands should pass explicit local overrides (or a separate local-only env file) rather than relying on the auto-loaded `.env`.

---

# 8. What Phase 18 Explicitly Does NOT Do

```text
new monitoring/reporting logic          → Phase 17 (reused as-is here)
model accuracy / ground-truth comparison → Phase 12 (isolated holdout, uses labels)
alert routing / paging                   → out of scope
automatic retraining trigger             → out of scope
CI wiring of this script                 → Phase 19 (Tests + CI)
```

---

# 9. Suggested Commit Sequence

```text
1. chore: pin requests for live-endpoint drift simulation
2. feat: add drift scenario payload builders
3. feat: add config.API_BASE_URL
4. feat: add simulate_drift.py CLI, reusing Phase 17's reporting pipeline
5. test: add drift scenario tests
6. docs: add phase 18 drift simulation doc
```

---

# 10. Next Phase

```text
Phase 19
=
tests + CI (pytest, ruff, package import, optional Docker build)
covering raw loader, delimiter handling, RUL, feature calculations,
data contracts, leakage boundaries, grouped CV, metrics,
model smoke test, API contract
```
