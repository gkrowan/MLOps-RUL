# Phase 5 — Dataset Boundary Refactor

## Goal

Make it structurally difficult for official holdout ground truth to leak into training, feature selection, preprocessing, CV, or AutoML.

## Production APIs

### Training

```python
build_training_dataset(TRAINING_SET_DIR)
```

Reads only `Training_set` and returns Feature Set V1 + `rul_seconds`.

### Official holdout inference features

```python
build_test_feature_dataset(TEST_SET_DIR)
```

Reads only `Test_set` and never reads `Validation_Set`. The output must not contain `rul_seconds`.

### Official holdout ground truth

```python
build_test_ground_truth(TEST_SET_DIR, VALIDATION_SET_DIR)
```

This is the only production API allowed to use `Validation_Set`. It produces only evaluation keys + `rul_seconds`; it does not create model features.

## Processed artifacts

```text
data/processed/
├── train_features.parquet
├── test_features.parquet
├── test_ground_truth.parquet
└── feature_schema.json
```

## Leakage rules

The default model input is signal features only.

Blocked predictors:

```text
split
bearing
elapsed_time_seconds
file_index
rul_seconds
```

`condition` is known at inference time and may be tested later as an explicit context feature, but it is not part of the default Feature Set V1 model input.

## Run

Training only — proves no Validation_Set dependency:

```bash
python scripts/build_datasets.py --mode train
```

Holdout features only — also proves no Validation_Set dependency:

```bash
python scripts/build_datasets.py --mode test-features
```

Ground truth separately:

```bash
python scripts/build_datasets.py --mode ground-truth
```

Or all artifacts:

```bash
python scripts/build_datasets.py --mode all
```

Validate boundaries:

```bash
python scripts/verify_processed_data.py
```

## Acceptance gate

- `rul_seconds` exists in `train_features.parquet`.
- `rul_seconds` does not exist in `test_features.parquet`.
- Ground-truth keys exactly match Test-set feature keys.
- Training rows contain only `Training_set`.
- Holdout feature rows contain only `Test_set`.
- Training and official holdout bearing IDs do not overlap.
- Blocked metadata/target columns are absent from the default model feature list.
- Training/test feature generation can run without `Validation_Set`.
