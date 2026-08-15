# MLOps-RUL — End-to-End MLOps Pipeline
## Reuse-First Phasewise Professional Rollout Plan

**Working branch:** create a phase-specific branch from latest `main`  
**Target branch:** `main` via Pull Request  
**Project:** FEMTO/PRONOSTIA Bearing Remaining Useful Life (RUL) Prediction

---

## 1. Engineering Objective

Build the complete ML/MLOps application layer from the original FEMTO raw data through production-style deployment and monitoring.

The project should demonstrate that the ML system is:

- reproducible
- versioned
- automated
- testable
- deployable
- observable
- protected against data leakage

Target lifecycle:

```text
Raw FEMTO Data
      ↓
DVC + MinIO
      ↓
Raw Data Validation
      ↓
Deterministic Feature Engineering
      ↓
Versioned Processed Datasets
      ↓
Bearing-Level Cross Validation
      ↓
Baseline Models
      ↓
AutoML / Hyperparameter Optimization
      ↓
MLflow Experiment Tracking
      ↓
MLflow Model Registry
      ↓
Airflow Orchestration
      ↓
FastAPI + Docker
      ↓
Prediction Logging
      ↓
Grafana + Evidently
      ↓
Drift Simulation
      ↓
CI + Reproducibility
```

---

# 2. Core Engineering Decision: Reuse the Existing Codebase

We will **not rewrite the project from scratch**.

The existing repository already contains a useful and tested data foundation. The professional approach is:

```text
Existing working components
        ↓
Verify them
        ↓
Preserve tested behavior
        ↓
Fix architectural weaknesses
        ↓
Productionize incrementally
```

## Components to reuse

| Existing component | Decision | Reason |
|---|---|---|
| `src/femto_rul/ingestion/raw_loader.py` | KEEP | Correctly handles per-file comma/semicolon delimiter differences |
| `src/femto_rul/labeling/rul.py` | KEEP CORE LOGIC | RUL formula is correct; ground-truth access needs isolation |
| `src/femto_rul/features/time_domain.py` | KEEP AS FEATURE SET V1 | Good simple baseline |
| `src/femto_rul/features/frequency_domain.py` | KEEP AS FEATURE SET V1 | Reasonable generic FFT-band features |
| `scripts/verify_data.py` | KEEP + REFACTOR | Strong dataset integrity checks |
| Existing unit tests | KEEP + EXPAND | Good foundation for CI |
| EDA notebooks | KEEP | Exploration/documentation only |
| Chris's Airflow/MLflow/MinIO/Grafana stack | REUSE | Infrastructure already works |
| Existing `pipeline.py` | REFACTOR | Current combined dataset creates leakage risk |
| Existing `features.parquet` | REFERENCE ONLY | Combined labeled artifact should not drive production training |

---

# 3. Known Issues to Fix

## 3.1 Combined dataset leakage path

Current production-like extraction builds:

```text
Training_Set
+
Validation_Set
+
Test_Set with true RUL derived from Validation_Set
```

into one `features.parquet`.

That artifact is useful for analysis but unsafe as a training input.

Target:

```text
train_features.parquet
test_features.parquet
test_ground_truth.parquet
```

with strict separation.

---

## 3.2 Dataset naming semantics

The released dataset names can be misleading for normal ML workflows.

Use these roles internally:

| Released name | MLOps role |
|---|---|
| `Training_set` | development/training data |
| `Test_set` | official holdout inference inputs |
| `Validation_Set` | ground-truth full trajectories for Test_set |

Internal validation must be constructed only from `Training_set`.

---

## 3.3 Random snapshot splits are prohibited

Snapshots from one bearing are temporally related.

Do not use ordinary row-level:

```python
train_test_split(...)
```

Use Leave-One-Bearing-Out / grouped cross-validation.

---

## 3.4 EDA and production feature definitions must agree

The current production implementation uses **Fisher/excess kurtosis**:

```text
Gaussian ≈ 0
```

Earlier EDA used Pearson-style interpretation:

```text
Gaussian ≈ 3
```

We will standardize the project definition and document it.

---

## 3.5 Dependency mismatch must be resolved before real training deployment

The inspected environments currently use different ML package versions.

Example observed mismatch:

```text
Application:
pandas 3.0.5
scikit-learn 1.9.0

Airflow training image:
pandas 2.1.4
scikit-learn 1.6.1
```

We must align the real training/serving dependency contract before serializing production models across environments.

---

# 4. Infrastructure Boundary

Chris owns infrastructure setup.

Expected locally forwarded services:

| Service | Local address | Use |
|---|---|---|
| Airflow | `http://localhost:8080` | orchestration |
| MLflow | `http://localhost:5000` | experiment tracking + registry |
| Grafana | `http://localhost:3000` | operational monitoring |
| MinIO Console | `http://localhost:9001` | storage administration |
| MinIO S3 API | normally `http://localhost:9000` | DVC / S3 clients |

Important:

```text
9001 = MinIO browser console
9000 = MinIO S3 API
```

The application layer should consume this infrastructure, not duplicate it.

---

# 5. Branch Strategy

The team is actively merging parallel work, so each implementation phase starts from the latest
`main` rather than keeping one long-running branch indefinitely.

Example:

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/dataset-boundary-pipeline
```

Confirm:

```bash
git branch --show-current
git status --short
```

Keep infrastructure, data-pipeline, modeling, and serving changes reviewable through focused pull
requests. After another team PR lands, sync from `main` before beginning the next phase.

---

# 6. Target Data Layout

The released FEMTO files are already available as extracted bearing directories, so the
project does not introduce an unnecessary `interim/` layer.

```text
data/
├── raw/
│   ├── Training_set/              # DVC-managed immutable source
│   ├── Test_set/                  # DVC-managed immutable source
│   ├── Validation_Set/            # DVC-managed official ground truth source
│   ├── Training_set.dvc           # Git-tracked DVC metadata
│   ├── Test_set.dvc
│   └── Validation_Set.dvc
│
└── processed/
    ├── train_features.parquet
    ├── test_features.parquet
    ├── test_ground_truth.parquet
    └── feature_schema.json
```

Rules:

```text
data/raw       = immutable source directories managed by DVC/MinIO
data/processed = generated ML-ready artifacts managed by the DVC pipeline
```

---

# 7. Dataset Contract

## Training_set

Allowed for:

- EDA
- feature engineering decisions
- grouped CV
- baseline training
- AutoML
- hyperparameter tuning
- final candidate training

## Test_set

Allowed only for:

- final feature generation
- deployed-model inference
- production validation

Do not use test distributions to tune modeling decisions.

## Validation_Set

Allowed only for:

- building official test ground truth
- final evaluation

Training/AutoML code must work without access to it.

---

# PHASE 1 — Foundation, Baseline and Reuse-First Refactor

**Detailed implementation document:** `docs/phase_01_foundation_and_reuse.md`

## Objective

Prepare the repository for professional MLOps development without changing valid ML behavior.

## Scope

- create branch
- establish baseline
- document reusable components
- standardize directory/configuration conventions
- remove hard-coded data-layout assumptions
- fix `.gitignore` so DVC metadata can be committed later
- define service/environment configuration
- align Python version contract
- document dependency mismatch
- preserve all existing tests
- define dataset roles before DVC begins

## Deliverables

```text
docs/e2e_mlops_pipeline_phasewise_rollout.md
docs/phase_01_foundation_and_reuse.md
docs/data_contract.md              # if implemented in Phase 1
.env.example                       # application-facing additions
src/femto_rul/config.py            # path/config refactor
.gitignore                         # DVC-friendly data rules
```

## Acceptance gate

Before Phase 2:

- all current tests still pass
- repository imports successfully
- no raw/processed data is committed directly
- data locations are centralized in configuration
- Training/Test/Validation roles are documented
- current feature code is unchanged unless required for consistency
- current raw loader behavior remains unchanged
- existing verifier still works on the local dataset

---

# PHASE 2 — DVC + MinIO Raw Data Versioning

## Objective

Version the immutable source directories with DVC and store the data remotely in MinIO.

## Tasks

- install `dvc[s3]`
- initialize DVC
- configure MinIO remote
- track the three raw source directories
- push DVC cache to MinIO
- verify `dvc pull` from a clean state
- never commit raw CSV content or `.dvc/cache` to Git

Example:

```bash
dvc init

dvc add data/raw/Training_set
dvc add data/raw/Test_set
dvc add data/raw/Validation_Set

dvc remote add -d minio s3://mlops-rul-dvc
dvc remote modify minio endpointurl http://localhost:9000

dvc push
```

Credentials must remain outside Git.

## Acceptance gate

A clean clone with MinIO credentials can reproduce the exact raw source data using:

```bash
dvc pull
```

---

# PHASE 3 — Raw Data Contract Gate

## Objective

Validate the DVC-reproduced source directories before any feature/model stage runs.

Target:

```text
dvc pull
   ↓
data/raw/{Training_set,Test_set,Validation_Set}
   ↓
python scripts/verify_data.py
   ↓
validated immutable source data
```

## Reuse

Keep the existing production components:

```text
scripts/verify_data.py
src/femto_rul/ingestion/raw_loader.py
```

The verifier distinguishes critical acceleration/data-contract failures from
non-blocking temperature warnings and exits non-zero on blocking failures.

---

# PHASE 4 — Production Feature Set V1

## Objective

Turn the existing tested feature code into the authoritative production feature implementation.

Reuse:

```text
src/femto_rul/features/time_domain.py
src/femto_rul/features/frequency_domain.py
```

Feature Set V1 begins with the existing features:

```text
Horizontal:
- RMS
- Fisher kurtosis
- skewness
- crest factor
- FFT band 0..7

Vertical:
- RMS
- Fisher kurtosis
- skewness
- crest factor
- FFT band 0..7
```

Do not add many new features before measuring V1.

Feature Set V1 is versioned through the authoritative implementation plus
`src/femto_rul/features/schema.py` / `feature_schema.json`. Do not add a second unused feature
configuration file until a pipeline parameter is genuinely varied.

Possible V2 improvements later:

- std
- peak-to-peak
- absolute peak
- spectral centroid
- dominant frequency
- relative band energy
- spectral entropy

Only add them after V1 baseline evidence.

---

# PHASE 5 — Dataset Boundary Refactor

## Objective

Remove the combined labeled dataset from the production training path.

Replace production use of:

```text
build_full_dataset()
```

with explicit APIs:

```text
build_training_dataset()
build_test_feature_dataset()
build_test_ground_truth()
```

Outputs:

```text
data/processed/
├── train_features.parquet
├── test_features.parquet
├── test_ground_truth.parquet
└── feature_schema.json
```

Hard rules:

```python
assert "rul_seconds" in train_features.columns
assert "rul_seconds" not in test_features.columns
```

Training code must run with `Validation_Set` unavailable.

The old combined `features.parquet` may be retained locally as a historical/reference artifact but must not be used for production training.

---

# PHASE 6 — DVC Processed Data Pipeline

## Objective

Make raw validation and processed feature/ground-truth generation reproducible.

Add:

```text
dvc.yaml
```

Feature Set V1 choices are fixed in version-controlled source. A separate `params.yaml`
should be introduced only when a pipeline parameter is genuinely varied (for example a
future feature-set/model experiment), rather than adding unused configuration.

Target stages:

```text
verify_raw
   ↓
build_processed
   ↓
verify_processed
```

Run:

```bash
dvc repro
dvc push
```

Acceptance:

Deleting generated `data/processed/*` and rerunning `dvc repro` reproduces the four expected
processed artifacts and passes the leakage/data-contract checks.

---

# PHASE 7 — Bearing-Level Cross Validation

## Objective

Measure generalization to unseen bearings.

Use Leave-One-Bearing-Out CV across the six training bearings.

Never split snapshots from the same bearing between training and validation.

Create:

```text
src/femto_rul/modeling/cv.py
tests/test_cv.py
```

Acceptance:

Every fold has zero bearing overlap between train and validation.

---

# PHASE 8 — Evaluation Metrics + Baseline Models

## Objective

Establish transparent reference performance before AutoML.

Metrics:

```text
Primary: RMSE

Secondary:
- MAE
- R²
- PHM12 asymmetric score
```

Models:

1. median-RUL baseline
2. Ridge/Linear Regression
3. Random Forest
4. XGBoost
5. LightGBM

Create:

```text
src/femto_rul/evaluation/metrics.py
src/femto_rul/modeling/baselines.py
scripts/train_baselines.py
```

Never include direct lifecycle leakage fields as predictors:

```text
rul_seconds
elapsed_time_seconds
file_index
bearing
split
```

Operating condition can be tested as a legitimate predictor.

---

# PHASE 9 — MLflow Experiment Tracking

## Objective

Make every meaningful experiment auditable.

Development tracking URI:

```text
http://localhost:5000
```

Log:

### Parameters
- model
- hyperparameters
- feature-set version
- random seed
- CV method

### Reproducibility
- Git commit SHA
- DVC data revision
- Python version
- important package versions

### Metrics
- fold RMSE
- mean/std CV RMSE
- MAE
- R²
- PHM score

### Artifacts
- fold predictions
- residual plots
- prediction-vs-actual
- feature importance
- feature schema
- model

---

# PHASE 10 — Controlled AutoML

## Objective

Automate model/hyperparameter selection without giving up validation control.

Recommended:

```text
Optuna
```

Candidates:

- Random Forest
- XGBoost
- LightGBM

Objective:

```text
mean Leave-One-Bearing-Out CV RMSE
```

Every trial logs to MLflow.

AutoML must never access Test_set ground truth.

---

# PHASE 11 — Final Candidate + Model Registry

## Objective

Train the selected configuration using the full Training_set and register it.

Flow:

```text
Best grouped-CV configuration
        ↓
Train on Training_set
        ↓
MLflow final run
        ↓
Model Registry
        ↓
femto-rul-model
```

Use aliases such as:

```text
candidate
champion
```

Record model signature and feature schema.

---

# PHASE 12 — Isolated Holdout Evaluation

## Objective

Evaluate the frozen registered candidate on official holdout inputs.

Flow:

```text
test_features.parquet
      ↓
registered candidate
      ↓
predictions
      ↓
separate evaluator
      +
test_ground_truth.parquet
      ↓
final metrics
```

Report:

- RMSE
- MAE
- R²
- PHM score
- per-bearing error

Do not tune the model from this result.

---

# PHASE 13 — Airflow Integration

## Objective

Orchestrate working application commands.

Airflow must call application code; it must not contain core ML logic.

Training DAG:

```text
validate_raw_data
        ↓
build_features
        ↓
validate_processed_data
        ↓
train_baselines
        ↓
run_automl
        ↓
train_final_candidate
        ↓
register_model
```

Reuse Chris's infrastructure smoke-test DAG as proof of connectivity, then replace synthetic training with real application entry points.

---

# PHASE 14 — FastAPI Model Serving

## Objective

Serve the registered model through a stable inference contract.

Create:

```text
api/
├── main.py
├── schemas.py
└── model_loader.py
```

Endpoints:

```text
GET  /health
GET  /model-info
POST /predict
```

The API must load the registered model, not a manually copied pickle.

Training and inference must share the same feature schema.

---

# PHASE 15 — Docker Application Image

## Objective

Make inference reproducible outside a developer laptop.

Add:

```text
Dockerfile
.dockerignore
```

Pin compatible ML dependencies.

Acceptance:

A clean Docker build starts FastAPI, reaches MLflow, loads the model, and answers `/health`.

---

# PHASE 16 — Prediction Logging + Grafana

## Objective

Monitor operational behavior.

Log per request:

```text
timestamp
request_id
model version
feature-set version
prediction
latency
status
```

Grafana should expose:

- request count
- latency
- error rate
- throughput
- active model version

---

# PHASE 17 — Evidently Data/Model Monitoring

## Objective

Monitor production input and prediction distributions against the training baseline.

Track:

- feature drift
- missing values
- out-of-range features
- prediction distribution
- schema changes

Use training features as the reference distribution.

---

# PHASE 18 — Drift Simulation

## Objective

Demonstrate monitoring response to abnormal production inputs.

Create:

```text
scripts/simulate_drift.py
```

Scenarios:

### Amplitude drift
Increase vibration-related magnitude/energy.

### Channel drift
Swap or systematically alter horizontal/vertical feature groups.

### Data-quality anomaly
Missing values, invalid schema, extreme values.

Save monitoring evidence for the presentation.

---

# PHASE 19 — Tests + CI

## Objective

Protect the MLOps pipeline during integration.

Tests:

- raw loader
- delimiter handling
- RUL
- feature calculations
- data contracts
- leakage boundaries
- grouped CV
- metrics
- model smoke test
- API contract

GitHub Actions:

```text
pytest
ruff
package import
optional Docker build
```

---

# PHASE 20 — Documentation + Reproducibility

## Objective

A new engineer should be able to reproduce the system from Git + DVC + credentials.

README must explain:

1. problem
2. dataset contract
3. architecture
4. environment
5. DVC pull
6. validation
7. features
8. training
9. AutoML
10. MLflow
11. Airflow
12. registry
13. API
14. monitoring
15. drift
16. final evaluation
17. tests

---

# PHASE 21 — End-to-End Dry Run

## Definition of Done

Demonstrate:

```text
1. Git identifies code version
2. DVC identifies data version
3. MinIO stores data/artifacts
4. raw validation passes
5. feature generation is reproducible
6. bearing-level validation prevents leakage
7. baseline models are documented
8. AutoML selects a configuration
9. MLflow records experiments
10. Model Registry versions the candidate
11. Airflow executes the workflow
12. FastAPI serves the registered model
13. Docker reproduces serving
14. Grafana shows operational metrics
15. Evidently establishes a clean baseline
16. simulated drift is detected
17. final test metrics are isolated and reproducible
```

---

# 8. Recommended Commit Sequence

Use small logical commits even though development is on one branch.

```text
1. docs: add reuse-first e2e rollout and phase 1 plan
2. refactor: centralize data and service configuration
3. chore: prepare DVC-friendly data layout
4. feat: version raw FEMTO archives with DVC
5. feat: add reproducible raw extraction
6. feat: productionize raw data validation
7. feat: formalize feature set v1
8. refactor: isolate train and holdout datasets
9. feat: add reproducible DVC feature pipeline
10. feat: add leave-one-bearing-out validation
11. feat: establish RUL model baselines
12. feat: add MLflow experiment tracking
13. feat: add grouped AutoML search
14. feat: register best RUL model
15. feat: add isolated holdout evaluation
16. feat: orchestrate training with Airflow
17. feat: serve registered model with FastAPI
18. feat: containerize inference service
19. feat: add inference telemetry
20. feat: add Evidently monitoring
21. feat: add reproducible drift simulation
22. ci: add automated quality gates
23. docs: finalize reproducible MLOps workflow
```

---

# 9. Immediate Execution

Do **Phase 1 first**.

Do not start AutoML or rewrite feature engineering yet.

Phase 1 should leave us with:

```text
Known-good existing foundation
        ↓
clean branch
        ↓
clear data contract
        ↓
central configuration
        ↓
DVC-ready repository structure
        ↓
all existing tests still passing
```

Then Phase 2 starts raw-data versioning.
