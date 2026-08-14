# FEMTO/PRONOSTIA Data Contract

## Purpose

This document defines how each released dataset split may be used by the MLOps pipeline. The contract exists to prevent target leakage and to make training/evaluation behavior explicit.

## Split roles

| Released dataset | MLOps role | Allowed usage |
|---|---|---|
| `Training_set` | Development/training trajectories with known failure endpoint | EDA, feature engineering, grouped CV, baseline training, AutoML, hyperparameter tuning, final candidate training |
| `Test_set` | Truncated official holdout trajectories | Final feature generation and inference only |
| `Validation_Set` | Full trajectories corresponding to `Test_set` bearings | Final holdout ground-truth derivation and scoring only |

## Leakage boundary

Model development must not use `Validation_Set` or true Test-set RUL for model selection, feature selection, preprocessing fitting, hyperparameter optimization, or AutoML.

Training code must be able to run when `Validation_Set` and final test ground truth are unavailable.

## Internal validation

Internal validation is created only from `Training_set` using bearing-level grouping. Snapshots from the same bearing must never be split between training and validation folds.

The planned default strategy is Leave-One-Bearing-Out cross-validation across the six training bearings.

## Processed dataset contract

The production pipeline will generate separate artifacts:

```text
data/processed/
├── train_features.parquet
├── test_features.parquet
├── test_ground_truth.parquet
└── feature_schema.json
```

Required rules:

- `train_features.parquet` contains training features and `rul_seconds`.
- `test_features.parquet` contains holdout features and **must not** contain `rul_seconds` or another true-future label.
- `test_ground_truth.parquet` is accessible only to the final evaluation path.
- The legacy combined `features.parquet` is reference/analysis-only and is not an approved AutoML training artifact.

## Feature Set V1

The existing production feature implementation is retained initially:

- horizontal/vertical RMS
- horizontal/vertical Fisher/excess kurtosis
- horizontal/vertical skewness
- horizontal/vertical crest factor
- 8 FFT-band-energy features per channel

Kurtosis convention for production Feature Set V1 is Fisher/excess kurtosis, where a Gaussian distribution is approximately `0`.
