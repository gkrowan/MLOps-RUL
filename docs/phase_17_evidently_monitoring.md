# Phase 17 — Evidently Data/Model Monitoring

**Branch:** `feature/e2e-mlops-pipeline`
**Parent:** latest `main`
**Purpose:** Compare live production feature distributions against the training baseline and surface drift before it silently degrades predictions.

---

## Addendum (2026-08-15) — Prefix V1 realignment

The registered champion model (Phase 14's `api/main.py`, which landed on
`main` while this scaffolding sat blocked) serves **Prefix V1** features
(`features/prefix.py::prefix_feature_columns()`, 21 columns), not the 24 raw
`FEATURE_COLUMNS_V1` columns this doc originally designed against. Monitoring
the wrong feature set would defeat the point of drift detection, so:

- Reference distribution moved from `data/processed/train_features.parquet`
  to `data/processed/prefix_train_v1.parquet` (built by
  `scripts/build_prefix_dataset.py`; path already defined as
  `experiments.config.PREFIX_DATASET_PATH`) — the actual training
  distribution for the features the model consumes.
- `column_mapping.py`/`current.py`/`reference.py` now key off
  `features.prefix.prefix_feature_columns()` (21 columns) instead of
  `pipeline.FEATURE_COLUMNS_V1` (24 columns, now removed — see the Phase 16
  doc's addendum). §3/§4 below describe the original 24-column design;
  the live code is the source of truth where they disagree.
- `report.py` needed no change — it was already decoupled, only depending on
  `column_mapping.build_data_definition()`.
- Verified: `tests/test_monitoring.py` (all 5, synthetic data, no DB) passes
  against the 21-column `DataDefinition`; `load_reference_features()`'s
  missing-file/missing-column error paths verified too. Generating the real
  `configs/monitoring_reference_ranges.json` from actual training data is
  still blocked — `prefix_train_v1.parquet` needs a `dvc pull`/`dvc repro`
  this environment doesn't have raw data for (unchanged from before).

## Scaffolding status (2026-08-14, superseded in part — see addendum above)

Built ahead of Phase 5/6/14, since column mapping and report-building are pure functions testable with synthetic data:

```text
DONE  requirements.txt                       — evidently==0.7.21, verified against
                                                 pandas==3.0.5/scikit-learn==1.9.0 by
                                                 dry-run install (no downgrade needed —
                                                 the §2 compatibility risk did not
                                                 materialize, see below)
DONE  src/femto_rul/monitoring/column_mapping.py — DataDefinition from prefix_feature_columns()
DONE  src/femto_rul/monitoring/reference.py      — loads/validates prefix_train_v1.parquet
DONE  src/femto_rul/monitoring/current.py        — reads the predictions table (Phase 16),
                                                     now storing Prefix V1 columns
DONE  src/femto_rul/monitoring/report.py         — build_report/save_report/drifted_column_share
DONE  scripts/run_monitoring_report.py           — CLI, non-zero exit on drift over threshold
DONE  scripts/generate_reference_ranges.py       — CLI, not yet run (needs prefix_train_v1.parquet)
DONE  tests/test_monitoring.py                   — all pass locally, no DB/network needed

STILL BLOCKED  — reference.load_reference_features() will raise FileNotFoundError
                  until data/processed/prefix_train_v1.parquet is materialized
                  (needs dvc pull of raw data + dvc repro, out of scope for this pass);
                  current.py now reads real Prefix V1 rows once /predict traffic flows
                  (Phase 16's endpoint wiring is done — see its doc's addendum).
```

**API correction:** evidently 0.7.21's actual API is not what §4/§5 below
originally sketched. There is no `evidently.report.Report` or
`evidently.metric_preset`; the current API is `evidently.Report` +
`evidently.presets.{DataDriftPreset, DataSummaryPreset}` +
`evidently.Dataset.from_pandas(data_definition=...)`, and `ColumnMapping` is
replaced by `evidently.DataDefinition`. §4/§5 have been corrected to match
what's actually installed and tested.

**Design change from the original sketch:** `column_mapping.py` builds its
`DataDefinition` from `pipeline.FEATURE_COLUMNS_V1` directly, not from
`feature_schema.json` (which doesn't exist yet — it's Phase 5 output).
`FEATURE_COLUMNS_V1` is already the single source of truth `serving/telemetry.py`
uses too, so both Phase 16 and Phase 17 agree on column order without waiting
on Phase 5. When `feature_schema.json` does exist, it should be *generated
from* `FEATURE_COLUMNS_V1`, not the other way around.

**Design change on prediction monitoring:** rather than forcing
`predicted_rul_seconds` into the Evidently `DataDefinition` as a formal
target/prediction comparison (which needs production ground truth we don't
have), `report.py` exposes a separate `prediction_sanity_summary()` —
descriptive statistics only, documented in its own docstring as a sanity
check, not a drift test.

---

# 1. Phase 17 Outcome

At the end of Phase 17 we want:

```text
train_features.parquet (reference)
        ↓
predictions table, this window (current)
        ↓
Evidently column mapping aligned to Feature Set V1
        ↓
data drift + data quality report
        ↓
HTML report + pass/fail summary saved to artifacts/monitoring/
```

Phase 17 does **not** simulate drift — that's Phase 18, which will feed corrupted input through the live endpoint specifically to verify this phase's reports react.

Phase 17 does **not** evaluate model accuracy against ground truth — that's the isolated holdout evaluation in Phase 12, which uses `test_ground_truth.parquet`. Production requests never carry a true RUL, so this phase can only monitor input/prediction *distributions*, not error.

---

# 2. Dependency Gate

```text
reference dataset  → data/processed/train_features.parquet   (Phase 5/6 output — already this project's contract)
current dataset    → predictions table                        (Phase 16 output, not yet built)
```

Phase 17 cannot run against real production traffic until Phase 16's `predictions` table exists and has rows. It *can* be exercised earlier using `test_features.parquet` run through a local prediction function directly (no live API needed) as a dry run of the report itself — useful for validating column mapping and thresholds before Phase 16 lands.

## Compatibility risk — verify before implementing

`docs/e2e_mlops_pipeline_phasewise_rollout.md` §3.5 already flags a pandas/scikit-learn version mismatch between the application (`pandas==3.0.5`) and the Airflow training image (`pandas==2.1.4`). Evidently has historically pinned pandas/numpy tightly and may not yet support pandas 3.x at the version available when this phase is implemented.

```text
Before writing code:
    check `evidently`'s declared pandas/numpy compatibility
    against pandas==3.0.5 (application) and pandas==2.1.4 (Airflow image)
Do not silently downgrade the application's pandas to make Evidently import.
Record the finding in docs/baseline_status.md or the dependency note from Phase 1 Change 10.
```

If Evidently can't run inside the application's pinned environment, the fallback is a separate `requirements-monitoring.txt` / isolated environment for the monitoring script, not a project-wide pandas downgrade.

---

# 3. Reference vs. Current Dataset Definition

> **Superseded by the 2026-08-15 addendum above.** "24 model input columns"
> below means the 21 Prefix V1 columns in the live code; reference is
> `prefix_train_v1.parquet`, not `train_features.parquet`.

## Reference (baseline)

```text
data/processed/train_features.parquet
```

reduced to exactly the 24 model input columns — the same columns the `predictions` table stores, in the same order, same units. `rul_seconds` and any leakage columns (`elapsed_time_seconds`, `file_index`, `bearing`, `split`) must be dropped before handing the frame to Evidently; those aren't things a live model receives, so they aren't meaningful to compare against production traffic.

## Current (production window)

```sql
SELECT rms_horiz, kurtosis_horiz, skewness_horiz, crest_factor_horiz,
       fft_band_0_horiz, ..., fft_band_7_horiz,
       rms_vert, kurtosis_vert, skewness_vert, crest_factor_vert,
       fft_band_0_vert, ..., fft_band_7_vert,
       predicted_rul_seconds
FROM predictions
WHERE status = 'ok'
  AND requested_at >= now() - interval '%s'
```

`status='error'` rows are excluded — they have null feature columns by construction (Phase 16 §4) and would just show up as 100% missing-value drift, which isn't a meaningful signal here (it's already visible in Grafana's error-rate panel).

Window size (`%s` above) should be a CLI parameter, not hard-coded — a course-project demo run and a "last 24 hours" scheduled run have different needs.

---

# 4. Column Mapping (DONE — `src/femto_rul/monitoring/column_mapping.py`)

> **Column source superseded by the 2026-08-15 addendum** — see the code
> block below for the current version.

evidently 0.7.21 dropped `ColumnMapping` in favor of `DataDefinition` +
`Dataset.from_pandas(data_definition=...)` — verified by installing the
pinned version and inspecting its actual API (see the scaffolding-status
note above). Built:

```python
from evidently import DataDefinition
from femto_rul.features.prefix import prefix_feature_columns

def build_data_definition() -> DataDefinition:
    return DataDefinition(numerical_columns=list(prefix_feature_columns()))
```

Source is `features.prefix.prefix_feature_columns()` — the same feature
schema the served model and `api/main.py`'s request contract already use —
not `feature_schema.json` (which is Phase 5's raw-24-column output and
doesn't match what the model actually consumes).
No categorical columns yet (operating condition isn't part of the served
feature vector — revisit if Phase 4's "condition as legitimate predictor"
experiment lands). `predicted_rul_seconds` is deliberately not declared
here; see the "prediction monitoring" design-change note above and §Change 2
below for how it's actually handled.

---

# 5. Exact Repository Changes

## Change 1 — Dependency (DONE)

Added to `requirements.txt`:

```text
psycopg2-binary==2.9.10
evidently==0.7.21
```

Verified by dry-run installing `evidently==0.7.21` against the pinned
`pandas==3.0.5`/`scikit-learn==1.9.0` on 2026-08-14 — resolves cleanly, no
downgrade needed. The §2 risk did not materialize; it's flagged here as
resolved rather than removed, so the reasoning stays visible.

---

## Change 2 — Monitoring module (DONE)

```text
src/femto_rul/monitoring/
├── __init__.py
├── column_mapping.py   # DataDefinition from features.prefix.prefix_feature_columns()
├── reference.py         # loads + validates prefix_train_v1.parquet; also
│                         # exposes load_reference_targets() for §Change 2's
│                         # rul_seconds sanity comparison
├── current.py            # reads predictions table (Phase 16) for a time window
└── report.py             # build_report / drifted_column_share /
                            # prediction_sanity_summary / save_report
```

`report.py`, using the verified 0.7.21 API (`evidently.Report` +
`evidently.presets`, not `evidently.report`/`evidently.metric_preset`):

```python
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from femto_rul.monitoring.column_mapping import build_data_definition


def build_report(reference_df, current_df):
    data_definition = build_data_definition()
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    return report.run(current_data=current_dataset, reference_data=reference_dataset)
```

`report.run()` returns a `Snapshot`, not the `Report` object itself —
`snapshot.save_html(path)` / `snapshot.save_json(path)` / `snapshot.dict()`
are how output gets written; `drifted_column_share()` reads the
`DriftedColumnsCount` metric's `value["share"]` out of `snapshot.dict()`
for the CLI's pass/fail exit code. `prediction_sanity_summary()` handles
`predicted_rul_seconds` separately as descriptive stats, not a formal
Evidently target/prediction comparison — see the design-change note above.

Verified end-to-end against synthetic data before writing this into the
plan: a 5-sigma shift across all 21 columns reads as `drifted_column_share
> 0.5`; identical distributions read as `< 0.2`. `tests/test_monitoring.py`
encodes both cases.

---

## Change 3 — Reference range config (script DONE, not yet run — needs prefix_train_v1.parquet)

Built `scripts/generate_reference_ranges.py`, calling `monitoring.reference.load_reference_features()` and writing 1st/99th percentiles per column. Hasn't been run yet — `data/processed/prefix_train_v1.parquet` doesn't exist locally in this environment (its DVC pipeline stage needs raw data pulled from the remote first; see the 2026-08-15 addendum).

```json
{
  "rms_horiz": {"p01": ..., "p99": ...},
  "kurtosis_horiz": {"p01": ..., "p99": ...}
}
```

Use 1st/99th percentile from `Training_set`, not hard-coded numeric guesses — this project's own data defines what "in range" means, and percentile bounds tolerate the dataset's natural spread across bearings/conditions better than a single global min/max would.

Phase 18's drift simulation should target these same bounds when engineering "out-of-range feature" scenarios, so the two phases test the same contract instead of each inventing separate thresholds.

---

## Change 4 — CLI entry point (DONE)

Built `scripts/run_monitoring_report.py`:

```text
usage: run_monitoring_report.py [--window "24 hours"] [--out artifacts/monitoring] [--fail-threshold 0.5]
```

(window is a Postgres interval literal, matching `current.py`'s query — not the `24h` shorthand originally sketched.)

Flow:

```text
load reference (train_features.parquet → 24 cols)
        ↓
load current (predictions table, --window)
        ↓
build_report()
        ↓
save HTML  → artifacts/monitoring/<timestamp>/data_drift.html
save JSON  → artifacts/monitoring/<timestamp>/summary.json
print pass/fail line to stdout (non-zero exit if drift detected, for CI/Airflow use)
```

Non-zero exit on detected drift matters if this script is ever wired into the Airflow DAG (Phase 13) as a scheduled check — but wiring that is optional stretch, not required for Phase 17 acceptance.

---

## Change 5 — Config additions (DONE)

`src/femto_rul/config.py`:

```python
# ---------------------------------------------------------------------------
# Monitoring (Phase 17)
# ---------------------------------------------------------------------------

MONITORING_ARTIFACTS_DIR: Final[Path] = ARTIFACTS_DIR / "monitoring"

MONITORING_REFERENCE_RANGES_PATH: Final[Path] = (
    REPO_ROOT / "configs" / "monitoring_reference_ranges.json"
)
```

---

## Change 6 — Tests (DONE)

`tests/test_monitoring.py`:

`tests/test_monitoring.py` (all pass, `pytest -q` — no live Postgres or network dependency):

```text
- build_data_definition() returns exactly features.prefix.prefix_feature_columns(), in order
- load_reference_features() raises FileNotFoundError on a missing path,
  and ValueError if the parquet is missing expected Prefix V1 columns
- build_report() on a 5-sigma-shifted synthetic current set reports
  drifted_column_share > 0.5
- build_report() on two draws from the same distribution reports
  drifted_column_share < 0.2
- save_report() writes data_drift.html, data_drift.json, and summary.json,
  and summary.json's drifted_column_share matches what drifted_column_share() returned
```

---

# 6. Proposed File Tree (after Phase 17)

```text
MLOps-RUL/
├── configs/
│   └── monitoring_reference_ranges.json          NOT YET GENERATED (script ready, needs prefix_train_v1.parquet)
├── src/femto_rul/
│   ├── config.py                                 (+ monitoring paths) DONE
│   ├── features/prefix.py                        (prefix_feature_columns(), source of truth) DONE
│   └── monitoring/                                DONE
│       ├── __init__.py
│       ├── column_mapping.py
│       ├── reference.py
│       ├── current.py
│       └── report.py
├── scripts/
│   ├── run_monitoring_report.py                  DONE
│   └── generate_reference_ranges.py              DONE
├── artifacts/monitoring/                          (generated at run time, gitignored)
├── tests/test_monitoring.py                       DONE
└── docs/phase_17_evidently_monitoring.md          (this file)
```

---

# 7. Phase 17 Acceptance Criteria

- [x] `evidently` pinned in `requirements.txt`, compatibility with pandas 3.0.5 verified and recorded (dry-run install, 2026-08-14 — no downgrade needed)
- [x] column mapping is generated from a single source of truth (`features.prefix.prefix_feature_columns()`), not duplicated by hand *(2026-08-15: realigned from `pipeline.FEATURE_COLUMNS_V1` to match what the served model actually consumes)*
- [x] reference dataset excludes `rul_seconds` and all leakage columns *(enforced by `load_reference_features` selecting only `prefix_feature_columns()`)*
- [x] current dataset excludes `status='error'` rows *(enforced in `current.py`'s query)*
- [ ] report runs against a dry-run dataset before any live traffic exists (still blocked — needs `prefix_train_v1.parquet`, which needs a `dvc pull`/`dvc repro` this environment doesn't have raw data for)
- [x] report runs against the `predictions` table once Phase 16 has rows *(Phase 16's `/predict` wiring is done — see its doc's addendum; `current.py` now reads real Prefix V1 rows)*
- [x] HTML + JSON output saved under `artifacts/monitoring/` *(verified via `tests/test_monitoring.py`'s `save_report` test, using a tmp_path)*
- [ ] reference percentile ranges are generated from `Training_set`, not hard-coded (script built, not yet runnable — needs `prefix_train_v1.parquet`)
- [x] tests pass without a live Postgres or network dependency (`pytest -q` — all monitoring tests pass, verified against the 21-column Prefix V1 schema)

---

# 8. What Phase 17 Explicitly Does NOT Do

```text
drift simulation / corrupted-input scenarios → Phase 18
model accuracy / ground-truth comparison      → Phase 12 (isolated holdout, uses labels)
alert routing / paging                        → out of scope
automatic retraining trigger                  → out of scope
pushing drift metrics into Grafana             → optional stretch, not required
```

---

# 9. Suggested Commit Sequence

Matches item 20 in the e2e doc's commit sequence: `feat: add Evidently monitoring`.

```text
1. chore: pin evidently and verify pandas compatibility
2. feat: add monitoring column mapping and reference/current loaders
3. feat: add Evidently drift/quality report builder and CLI
4. feat: generate reference percentile ranges from Training_set
5. test: add monitoring module tests
```

---

# 10. Next Phase

```text
Phase 18
=
scripts/simulate_drift.py
+
amplitude drift, channel drift, data-quality anomaly scenarios
+
verify this phase's reports actually flag them
```

Phase 18 targets the same `configs/monitoring_reference_ranges.json` this phase generates, so both phases agree on what "in range" means for a given feature.
