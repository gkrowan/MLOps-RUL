# Baseline Status — Before E2E MLOps Refactor

Fill this document from the local project checkout before behavioral refactoring.

## Source

- Branch: `feature/e2e-mlops-pipeline`
- Parent/main SHA: `<run: git rev-parse main>`
- Working SHA: `<run: git rev-parse HEAD>`

## Runtime

- Python: `<run: python --version>`
- OS: `<record local environment>`

## Existing tests

Command:

```bash
python -m pytest -q
```

Result:

```text
<paste result>
```

## Existing raw-data validation

Command:

```bash
python scripts/verify_data.py --data-dir <current extracted data path>
```

Result summary:

- Training bearings:
- Test bearings:
- Validation bearings:
- Integrity issues:

## Legacy feature artifact

If available, record without making it a production dependency:

- Path:
- Rows:
- Columns:
- Column names:

## Infrastructure connectivity

Run:

```bash
python scripts/check_environment.py
```

Record:

| Service | Status |
|---|---|
| Airflow | |
| MLflow | |
| Grafana | |
| MinIO Console | |
| MinIO S3 API | |

## Known integration issue

Before real model training is run in Airflow, align the ML dependency versions used by the application/training environment and Chris's Airflow image. Do not rely on cross-version model serialization.
