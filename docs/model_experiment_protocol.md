# Canonical FEMTO RUL Model Experiment Protocol

## Why this exists

Earlier exploratory work intentionally tested multiple formulations. From this point onward, model-family comparison and HPO use one frozen development benchmark so results are comparable and reproducible.

## Frozen development benchmark

- Source: `Training_set` only.
- Independent units: 6 complete run-to-failure bearings.
- Validation: Leave-One-Bearing-Out (LOBO), 6 folds.
- Representation: bearing prefixes built only from observations available up to the prediction time.
- Prefix grid: 40%, 45%, 50%, ..., 95% of each complete training trajectory.
- Canonical rows: 72 pseudo-prefix states.
- Target: direct RUL in seconds.
- Primary selection metric: mean LOBO RMSE.
- Secondary diagnostics: RMSE standard deviation, median RMSE, worst-bearing RMSE, MAE, R², PHM12 diagnostic, monotonic violation rate.
- `Test_set` and `Validation_Set`: not accessed during development/model selection.

## Time-series interpretation

Each bearing is its own degradation trajectory. Acceleration snapshots are recorded at roughly 10-second intervals. Bearings are **not** concatenated into one global time series. Prefix features summarize only the observed history of one bearing.

## DVC vs MLflow

DVC owns reproducible data inputs and feature/prefix datasets:

`raw -> train_features.parquet -> prefix_train_v1.parquet`

MLflow owns model experiments:

- model/hyperparameters
- Git SHA and hashes for `dvc.lock` / `params.yaml`
- LOBO metrics
- per-bearing metrics
- predictions and plots
- fitted development model artifact

The MLflow Model Registry is used only after a candidate has been selected. Registration does not imply final Test/Validation approval.

## Canonical experiment IDs

| ID | Model |
|---|---|
| E100 | condition life prior |
| E101 | median |
| E102 | ridge |
| E103 | KNN |
| E104 | random forest |
| E105 | ExtraTrees |
| E106 | HistGradientBoosting |
| E107 | XGBoost |
| E108 | LightGBM |

## Historical exploratory findings

These are retained for context but should not be mixed into the canonical scoreboard when their prefix grid/protocol differs.

- Snapshot median: mean RMSE ~5,556 sec.
- Snapshot HGB V1: ~6,524 sec.
- Temporal V2 HGB: ~8,061 sec.
- Prefix RF on older 55/65/75/85/95 grid: ~3,303 sec.
- Canonical expanded prefix direct-RUL RF: ~4,344 sec on the harder 40..95 grid.
- Total-life target variants were rejected because they were materially worse than direct RUL.

## Promotion sequence

1. `dvc repro` and `dvc push` to establish the canonical prefix dataset.
2. Run E100-E108 under the same benchmark and log every run to MLflow.
3. Select the best model family using LOBO metrics, emphasizing mean RMSE and worst-bearing RMSE.
4. Run Optuna only for the selected/competitive families, again using LOBO mean RMSE.
5. Register the selected full-Training_set model with alias `candidate`.
6. Only after the candidate is frozen, perform the one-time official `Test_set` inference and `Validation_Set` scoring phase.
