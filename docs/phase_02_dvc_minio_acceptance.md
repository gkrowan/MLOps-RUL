# Phase 2 — DVC + MinIO Acceptance Runbook

## Goal

Prove that Git metadata + DVC + the real cloud MinIO remote can reproduce the immutable FEMTO source dataset from a clean checkout.

## Preconditions

- `main` contains the three `.dvc` metadata files for `Training_set`, `Test_set`, and `Validation_Set`.
- The MinIO S3 API is reachable through the SSH tunnel at `http://127.0.0.1:9000`.
- The MinIO bucket `mlops-rul-dvc` exists.
- Real MinIO credentials are available locally through a secure environment source.

## 1. Verify the remote and local-only credentials

```bash
dvc remote list
cat .dvc/config
```

Expected shared config:

```ini
[core]
    remote = minio

['remote "minio"']
    url = s3://mlops-rul-dvc
    endpointurl = http://127.0.0.1:9000
```

Credentials must not appear in `.dvc/config`.

Load credentials locally, then store them only in DVC local config:

```bash
set -a
source .env
set +a

dvc remote modify --local minio access_key_id "$MINIO_ROOT_USER"
dvc remote modify --local minio secret_access_key "$MINIO_ROOT_PASSWORD"
```

Verify the secret-bearing file is ignored:

```bash
git check-ignore -v .dvc/config.local
```

Do not print `.dvc/config.local`.

## 2. Verify MinIO S3 health

```bash
curl -s -o /dev/null \
  -w "MinIO S3 API: %{http_code}\n" \
  http://127.0.0.1:9000/minio/health/live
```

Expected:

```text
MinIO S3 API: 200
```

## 3. Compare local DVC cache with cloud

```bash
dvc status
dvc status -c
```

Before the first upload it is normal for remote objects to be reported as missing/new.
Authentication or connectivity errors are not acceptable.

## 4. Push immutable source data

```bash
dvc push
```

Then:

```bash
dvc status -c
```

The local cache and `minio` remote should be in sync.

## 5. Clean-clone reproducibility test

Use a separate directory; do not delete the working dataset just to test recovery.

```bash
cd /tmp
rm -rf MLOps-RUL-dvc-acceptance
git clone git@github.com:gkrowan/MLOps-RUL.git MLOps-RUL-dvc-acceptance
cd MLOps-RUL-dvc-acceptance
git switch main
```

Create/activate the project environment and install requirements.

Configure only local DVC credentials in the clean clone:

```bash
dvc remote modify --local minio access_key_id "$MINIO_ROOT_USER"
dvc remote modify --local minio secret_access_key "$MINIO_ROOT_PASSWORD"
```

Pull:

```bash
dvc pull
```

Expected source directories:

```text
data/raw/Training_set/
data/raw/Test_set/
data/raw/Validation_Set/
```

Run the raw contract gate:

```bash
python scripts/verify_data.py
```

Expected final state: `PASS WITH WARNINGS` with zero critical acceleration/holdout-integrity issues.

## Phase 2 definition of done

- Raw directories are not tracked directly by Git.
- `.dvc/cache` and `.dvc/config.local` are ignored.
- `dvc push` succeeds against the real MinIO bucket.
- `dvc status -c` is synchronized.
- A clean clone can run `dvc pull` and reconstruct all three source directories.
- `scripts/verify_data.py` passes on the reconstructed data.
