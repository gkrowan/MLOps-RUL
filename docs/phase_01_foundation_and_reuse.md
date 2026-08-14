# Phase 1 — Foundation, Baseline and Reuse-First Refactor

**Branch:** `feature/e2e-mlops-pipeline`  
**Parent:** latest `main`  
**Purpose:** Prepare the existing FEMTO codebase for professional MLOps work without unnecessarily rewriting tested functionality.

---

# 1. Phase 1 Outcome

At the end of Phase 1 we want:

```text
Existing working data code
        ↓
verified baseline
        ↓
documented reuse decisions
        ↓
clean configuration
        ↓
professional data directory contract
        ↓
DVC-friendly Git rules
        ↓
service connection contract
        ↓
same existing tests still passing
```

Phase 1 does **not** train a new model.

Phase 1 does **not** implement AutoML.

Phase 1 does **not** replace the existing feature calculations.

It prepares a trustworthy foundation for DVC and the later production pipeline.

---

# 2. Existing Code Review — What We Are Keeping

The inspected repository already has useful production-style components.

## 2.1 Keep `src/femto_rul/ingestion/raw_loader.py`

Current strengths:

- loads one acceleration file at a time
- loads temperature files
- detects comma vs semicolon delimiter per file
- orders files by parsed file index
- handles missing temperature streams without crashing

Decision:

```text
KEEP
```

Phase 1 should not rewrite this module.

Later phases may add validation around it, but the core loader is reusable.

---

## 2.2 Keep `src/femto_rul/labeling/rul.py`

Current RUL formula:

```text
RUL(file_index)
=
(total_snapshots - file_index)
× 10 seconds
```

This is appropriate for complete run-to-failure trajectories.

Decision:

```text
KEEP CORE LABELING MATH
```

Important architectural change comes later:

```text
Training code
must not have access to
Test_set ground truth
```

The problem is not the formula itself; it is the current combined pipeline boundary.

---

## 2.3 Keep `src/femto_rul/features/time_domain.py`

Existing Feature Set V1:

```text
RMS
Fisher/excess kurtosis
skewness
crest factor
```

Decision:

```text
KEEP AS FEATURE SET V1
```

Important documentation rule:

```text
Fisher/excess kurtosis:
Gaussian ≈ 0
```

EDA and production code should use/document the same convention.

---

## 2.4 Keep `src/femto_rul/features/frequency_domain.py`

Current logic:

```text
25.6 kHz signal
      ↓
real FFT
      ↓
0 to 12.8 kHz
      ↓
8 equal-width frequency bands
      ↓
band energy features
```

Decision:

```text
KEEP AS FEATURE SET V1
```

Do not replace it yet with more complicated fault-frequency calculations unless the bearing geometry is verified.

---

## 2.5 Keep `scripts/verify_data.py`

This script already checks useful real-world dataset behavior:

- file numbering gaps
- acceleration shapes
- temperature shapes
- delimiter irregularities
- stray files
- Test_set truncation against Validation_Set

Decision:

```text
KEEP LOGIC
```

Later refactor it so reusable functions live under:

```text
src/femto_rul/validation/
```

and the script remains a thin CLI.

That refactor is Phase 3, not Phase 1.

---

## 2.6 Keep existing tests

Current tests cover:

- raw loader behavior
- time-domain feature calculations
- frequency features
- RUL calculations
- pipeline integration when local raw data exists

Decision:

```text
KEEP + EXPAND LATER
```

Phase 1's main requirement is that current tests do not regress.

---

# 3. Existing Code That Needs Refactoring Later

## `src/femto_rul/pipeline.py`

Current high-level flow can create:

```text
Training_set
+
Validation_Set
+
Test_set labeled from Validation_Set
```

inside one table.

Current function:

```text
build_full_dataset()
```

is useful for analysis but should not remain the production training boundary.

Do not fix this entire issue in Phase 1.

Document it now and refactor in the dedicated dataset-isolation phase.

Target later:

```text
build_training_dataset()
build_test_feature_dataset()
build_test_ground_truth()
```

---

# 4. Existing `features.parquet`

The existing file is useful for:

- checking current feature schema
- comparing future refactored output
- validating row counts
- regression comparison

It should **not** be treated as the authoritative future model-training dataset.

Reason:

```text
current combined artifact
contains labeled holdout data
```

Phase 1 rule:

```text
Keep locally as reference only.
Do not commit it to Git.
Do not use it as AutoML training input.
```

A later phase will regenerate:

```text
train_features.parquet
test_features.parquet
test_ground_truth.parquet
```

---

# 5. Phase 1 — Exact Repository Changes

## Change 1 — Create the branch

From a clean `main`:

```bash
git checkout main
git pull origin main
git switch -c feature/e2e-mlops-pipeline
```

Verify:

```bash
git branch --show-current
git status
```

Expected:

```text
feature/e2e-mlops-pipeline
```

---

# Change 2 — Add planning documents

Create:

```text
docs/
├── e2e_mlops_pipeline_phasewise_rollout.md
└── phase_01_foundation_and_reuse.md
```

Do not keep the project execution plan only at repository root.

The `docs/` folder becomes the source for architecture/engineering documentation.

Suggested commit:

```text
docs: add reuse-first e2e rollout and phase 1 plan
```

---

# Change 3 — Record the baseline before refactoring

Run:

```bash
python --version
python -m pytest -q
python scripts/verify_data.py
git rev-parse HEAD
```

If the current feature extraction is practical locally:

```bash
python scripts/extract_features.py
```

Record:

- Git SHA
- Python version
- test result
- number of detected bearings
- feature row count
- feature column count
- current feature column list
- local infrastructure URLs that are reachable

Create:

```text
docs/baseline_status.md
```

Example content:

```text
Branch point SHA:
Python:
Tests:
Raw validation:
Legacy features.parquet rows:
Legacy features.parquet columns:
MLflow reachable:
Airflow reachable:
Grafana reachable:
MinIO console reachable:
MinIO S3 API reachable:
```

Why:

Later, if a refactor changes behavior, we have a concrete reference.

---

# Change 4 — Formalize the data directory contract

## Current state

The existing config assumes:

```text
data/Training_set
data/Test_set
data/Validation_Set
```

and `.gitignore` ignores the entire:

```text
data/
```

## Target state

```text
data/
├── raw/
│   ├── Training_set.zip
│   ├── Test_set.zip
│   └── Validation_Set.zip
│
├── interim/
│   ├── Training_set/
│   ├── Test_set/
│   └── Validation_Set/
│
└── processed/
```

Meanings:

```text
raw       = immutable source archives
interim   = generated extracted raw files
processed = generated ML-ready tables
```

In Phase 1, create only the directory contract/placeholders.

Do not put large data into Git.

Suggested tracked placeholders:

```text
data/raw/.gitkeep
data/interim/.gitkeep
data/processed/.gitkeep
```

If DVC later replaces the need for some `.gitkeep` files, they can be removed.

---

# Change 5 — Refactor `src/femto_rul/config.py`

## Current config

Conceptually:

```python
REPO_ROOT = ...
DATA_DIR = REPO_ROOT / "data"

TRAINING_SET_DIR = DATA_DIR / "Training_set"
VALIDATION_SET_DIR = DATA_DIR / "Validation_Set"
TEST_SET_DIR = DATA_DIR / "Test_set"
```

## Target Phase 1 config

Introduce explicit layers:

```python
REPO_ROOT
DATA_ROOT
RAW_DATA_DIR
INTERIM_DATA_DIR
PROCESSED_DATA_DIR
ARTIFACTS_DIR
```

Then define:

```python
TRAINING_SET_DIR = INTERIM_DATA_DIR / "Training_set"
TEST_SET_DIR = INTERIM_DATA_DIR / "Test_set"
VALIDATION_SET_DIR = INTERIM_DATA_DIR / "Validation_Set"
```

Also define immutable archive locations:

```python
TRAINING_ARCHIVE
TEST_ARCHIVE
VALIDATION_ARCHIVE
```

Suggested shape:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
INTERIM_DATA_DIR = DATA_ROOT / "interim"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

TRAINING_ARCHIVE = RAW_DATA_DIR / "Training_set.zip"
TEST_ARCHIVE = RAW_DATA_DIR / "Test_set.zip"
VALIDATION_ARCHIVE = RAW_DATA_DIR / "Validation_Set.zip"

TRAINING_SET_DIR = INTERIM_DATA_DIR / "Training_set"
TEST_SET_DIR = INTERIM_DATA_DIR / "Test_set"
VALIDATION_SET_DIR = INTERIM_DATA_DIR / "Validation_Set"
```

Keep existing scientific constants:

```text
ACCELEROMETER_SAMPLING_RATE_HZ
ACC_SAMPLES_PER_FILE
ACC_COLUMNS
FILE_INTERVAL_SECONDS
CONDITIONS
```

Do not alter verified dataset constants without evidence.

---

# Change 6 — Support environment overrides only where useful

Do not turn every constant into an environment variable.

Stable dataset science/config belongs in Python/config files.

Environment-dependent service locations belong in environment variables.

Examples:

```text
MLFLOW_TRACKING_URI
MINIO_ENDPOINT_URL
DVC_REMOTE_BUCKET
MODEL_NAME
```

Avoid hard-coding:

```text
http://localhost:5000
http://localhost:9000
```

inside modeling modules.

---

# Change 7 — Update `.env.example`

Current `.env.example` mostly describes the Docker infrastructure.

Add application-facing configuration without real secrets.

Suggested additions:

```bash
# Application / ML clients
MLFLOW_TRACKING_URI=http://localhost:5000
MINIO_ENDPOINT_URL=http://localhost:9000
DVC_REMOTE_BUCKET=mlops-rul-dvc
MODEL_NAME=femto-rul-model
MODEL_ALIAS=candidate
```

MinIO credentials may be represented with placeholders:

```bash
AWS_ACCESS_KEY_ID=change-me
AWS_SECRET_ACCESS_KEY=change-me
```

Do not commit actual cloud credentials.

Document:

```text
localhost:9001 = console
localhost:9000 = S3 API
```

---

# Change 8 — Fix `.gitignore` for DVC

## Current problem

Current `.gitignore` contains:

```gitignore
data/
```

This is too broad for the planned DVC layout because it hides everything below `data/`, including metadata/placeholders we may need to track.

## Target principle

Ignore actual data content, not the entire data namespace.

For Phase 1, replace the blanket rule with something similar to:

```gitignore
# Raw local dataset content
data/raw/*.zip

# Generated extraction
data/interim/*
!data/interim/.gitkeep

# Generated processed datasets
data/processed/*
!data/processed/.gitkeep

# Generated artifacts
artifacts/*
!artifacts/.gitkeep
```

Important:

DVC will also maintain relevant `.gitignore` entries as files are added.

Do not manually ignore:

```text
*.dvc
dvc.yaml
dvc.lock
.dvc/
```

Those are Git metadata/config and must be versioned as appropriate.

---

# Change 9 — Align the Python version contract

Observed repository state:

```text
.python-version = 3.12
README = Python 3.12
pyproject.toml = >=3.11
```

Phase 1 should make this consistent.

Recommended project contract:

```text
Python 3.12
```

For example:

```toml
requires-python = ">=3.12,<3.13"
```

Why:

- the repository already standardizes on `.python-version` 3.12
- the README already instructs Python 3.12
- a narrow project runtime reduces environment drift during the course project

Do not change Chris's running infrastructure image blindly.

Record any incompatibility and align it during integration.

---

# Change 10 — Document dependency compatibility

Observed mismatch in the inspected branch:

```text
Application
-----------
pandas==3.0.5
scikit-learn==1.9.0

Airflow environment
-------------------
pandas==2.1.4
scikit-learn==1.6.1
```

Do not silently ignore this.

Create a short section in:

```text
docs/baseline_status.md
```

or:

```text
docs/integration_contract.md
```

stating that real model training/serving will use a single compatible dependency contract before Airflow production integration.

Phase 1 does not need to modify Chris's branch.

---

# Change 11 — Create `docs/data_contract.md`

Document the official use of each split.

Minimum content:

```text
Training_set
------------
Allowed:
EDA, CV, training, AutoML

Test_set
--------
Allowed:
final feature generation and inference only

Validation_Set
--------------
Allowed:
final test ground-truth derivation only
```

Add the key rule:

> Model selection, preprocessing fitting, feature selection and hyperparameter optimization must not use Test_set ground truth or Validation_Set.

Also document:

```text
Internal validation = Leave-One-Bearing-Out from Training_set
```

even though the actual CV implementation occurs later.

---

# Change 12 — Preserve Feature Set V1

Do not expand the feature set in Phase 1.

Existing authoritative features remain:

```text
rms_horiz
kurtosis_horiz
skewness_horiz
crest_factor_horiz
fft_band_0_horiz ... fft_band_7_horiz

rms_vert
kurtosis_vert
skewness_vert
crest_factor_vert
fft_band_0_vert ... fft_band_7_vert
```

Create a note:

```text
Feature Set Version: v1
Kurtosis convention: Fisher/excess
```

Later we may add configuration such as:

```text
configs/features_v1.yaml
```

but Phase 1 should avoid unnecessary numerical changes.

---

# Change 13 — Do Not Refactor `pipeline.py` Yet

The leakage issue is known, but Phase 1 should remain small and safe.

Do not combine configuration cleanup with a large behavior change.

Phase 1 documents:

```text
build_full_dataset() = legacy/analysis path
```

The dedicated later phase will replace it for production with:

```text
build_training_dataset()
build_test_feature_dataset()
build_test_ground_truth()
```

This separation makes code review easier.

---

# Change 14 — Keep the current EDA notebooks

Do not delete:

```text
notebooks/eda.ipynb
notebooks/chris-eda.ipynb
```

They remain exploratory evidence.

Rule going forward:

```text
Notebook = exploration
src/     = reusable production logic
scripts/ = command-line entry points
```

Any feature deemed production-worthy must move into `src/femto_rul/`.

---

# 6. Proposed Phase 1 File Tree

After Phase 1, the relevant structure should look like:

```text
MLOps-RUL/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── docs/
│   ├── e2e_mlops_pipeline_phasewise_rollout.md
│   ├── phase_01_foundation_and_reuse.md
│   ├── baseline_status.md
│   ├── data_contract.md
│   └── data_notes.md
│
├── notebooks/
│
├── scripts/
│   ├── extract_features.py
│   └── verify_data.py
│
├── src/femto_rul/
│   ├── config.py
│   ├── ingestion/
│   ├── labeling/
│   ├── features/
│   └── pipeline.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
└── requirements.txt
```

---

# 7. Phase 1 Tests

Run before changes:

```bash
python -m pytest -q
```

Run again after changes:

```bash
python -m pytest -q
```

Also test:

```bash
python -c "import femto_rul; from femto_rul import config; print(config.REPO_ROOT)"
```

If local raw data has already been moved into the new interim structure:

```bash
python scripts/verify_data.py --data-dir data/interim
```

If not, do not destructively move it just to satisfy Phase 1.

Phase 2/3 will formalize DVC + extraction.

---

# 8. Phase 1 Acceptance Criteria

Phase 1 is complete only when all of these are true:

- [ ] branch is `feature/e2e-mlops-pipeline`
- [ ] rollout document exists under `docs/`
- [ ] Phase 1 implementation document exists under `docs/`
- [ ] baseline status is recorded
- [ ] dataset roles are documented
- [ ] existing reusable modules are explicitly preserved
- [ ] Python runtime contract is consistent
- [ ] project data paths are centralized
- [ ] `.gitignore` is DVC-friendly
- [ ] `.env.example` documents application service endpoints
- [ ] no credentials are committed
- [ ] no raw ZIP/parquet data is committed directly
- [ ] existing tests pass
- [ ] existing loader/feature calculations have not been unnecessarily rewritten
- [ ] legacy combined `features.parquet` is documented as reference-only
- [ ] known dependency mismatch is documented for later infrastructure integration

---

# 9. Suggested Phase 1 Commit Sequence

Prefer several reviewable commits.

## Commit 1

```text
docs: add reuse-first e2e rollout and phase 1 plan
```

Files:

```text
docs/e2e_mlops_pipeline_phasewise_rollout.md
docs/phase_01_foundation_and_reuse.md
```

## Commit 2

```text
docs: define FEMTO dataset and leakage contract
```

Files:

```text
docs/data_contract.md
docs/baseline_status.md
```

## Commit 3

```text
refactor: centralize project data paths
```

Files:

```text
src/femto_rul/config.py
pyproject.toml
```

## Commit 4

```text
chore: prepare repository for DVC data versioning
```

Files:

```text
.gitignore
.env.example
data/raw/.gitkeep
data/interim/.gitkeep
data/processed/.gitkeep
```

Then:

```bash
python -m pytest -q
git status
```

---

# 10. What Phase 1 Explicitly Does NOT Do

Do not add these yet:

- DVC remote configuration
- MinIO credentials
- AutoML
- Optuna
- model training changes
- MLflow model logging
- Model Registry code
- Airflow production DAG
- FastAPI
- Evidently
- drift simulation
- large feature expansion
- rewrite of raw loader
- rewrite of feature modules
- final split refactor

This keeps the first PR/checkpoint easy to verify.

---

# 11. Next Phase

After Phase 1 is accepted:

```text
Phase 2
=
DVC initialization
+
MinIO S3 remote
+
raw ZIP tracking
+
dvc push/pull reproducibility
```

Only after the raw source is reproducibly versioned do we proceed to extraction and production data validation.
