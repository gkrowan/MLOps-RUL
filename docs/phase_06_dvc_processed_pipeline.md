# Phase 6 — DVC Processed Data Pipeline

## Goal

Make the path from DVC-reproduced raw directories to validated processed artifacts reproducible with one command.

## Pipeline

The supplied `dvc.yaml` contains:

```text
verify_raw
    ↓
build_processed
    ↓
verify_processed
```

### `verify_raw`

Runs:

```bash
python scripts/verify_data.py
```

It blocks on critical raw-data contract failures while known temperature irregularities remain warnings for Feature Set V1.

### `build_processed`

Runs:

```bash
python scripts/build_datasets.py --mode all
```

and creates:

```text
data/processed/train_features.parquet
data/processed/test_features.parquet
data/processed/test_ground_truth.parquet
data/processed/feature_schema.json
```

### `verify_processed`

Runs:

```bash
python scripts/verify_processed_data.py
```

and enforces the leakage boundary before modeling.

## First run

```bash
dvc repro
```

This creates/updates `dvc.lock`.

Review:

```bash
dvc status
git status --short
git diff -- dvc.yaml dvc.lock
```

Then push DVC cache/outputs to MinIO:

```bash
dvc push
```

## Reproducibility acceptance

After a successful push, generated processed artifacts should be recoverable/rebuildable from Git + DVC source data.

A safe local test is to remove only generated processed outputs (not raw data):

```bash
rm -f \
  data/processed/train_features.parquet \
  data/processed/test_features.parquet \
  data/processed/test_ground_truth.parquet \
  data/processed/feature_schema.json
```

Then:

```bash
dvc repro
python scripts/verify_processed_data.py
```

Do not delete the only copy of raw data for this test; Phase 2 already tests raw recovery through a clean `dvc pull`.
