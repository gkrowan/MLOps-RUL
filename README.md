# Project Proposal: Predictive Maintenance for Rolling-Element Bearings (FEMTO/PRONOSTIA Dataset)

Dataset available here (#10 in the list): https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Group Members: Anuradha Rakh, Chris Korabik, Erika Yan, Grace Rowan

Dataset: We plan to use the FEMTO Bearing Dataset (PRONOSTIA platform, IEEE PHM 2012 Challenge), which contains run-to-failure vibration data from 17 rolling-element bearings tested under three operating conditions (varying rotational speed and radial load). Each bearing is monitored via two accelerometers sampled at 25.6kHz, with readings captured every 10 seconds until failure. Failure is defined as the point where vibration amplitude exceeds a fixed 20g threshold. The dataset comes with a predefined train/test split: several bearings run to full failure (training), while others are deliberately truncated before failure, with true remaining life held out for evaluation.

Problem statement:  We will frame this as a regression problem predicting Remaining Useful Life (RUL) in seconds/cycles from a bearing's current vibration signal. Because raw waveforms are high-dimensional and non-stationary, our pipeline will extract time-domain (RMS, kurtosis, skewness, crest factor) and frequency-domain (FFT band energy) features per sample before modeling. Our primary evaluation metric will be RMSE on held-out truncated runs, supplemented by the PHM12 Challenge's asymmetric scoring function, which penalizes late predictions more heavily than early ones. This reflects the real-world cost asymmetry of late maintenance/unpredicted failures versus conservative, preventative maintenance. Our choice of model will likely be XGBoost or LightGBM using the extracted features, but we could potentially opt for a transfer learning or fine-tuning approach. 

Architecture/Tools:  For out pipeline we plan to use Airflow to orchestrate feature extraction and training as a DAG, MLflow to track experiments (model type, hyperparameters, RMSE/scoring metrics) and manage a versioned model registry, Docker and FastAPI to containerize the selected model for real-time inference, and EvidentlyAI to monitor incoming feature distributions against a training baseline. For our drift simulation, we can corrupt vibration inputs (e.g., inject out-of-range amplitude spikes or swap accelerometer channels) to verify the monitoring dashboard flags the resulting distributional shift.

See [femto_mlops_project_plan.md](femto_mlops_project_plan.md) for the week-by-week execution plan and team role breakdown.

## Repo structure

```
data/raw/               DVC-managed immutable FEMTO source directories
data/processed/         generated ML-ready feature/ground-truth artifacts
src/femto_rul/          installable package: config, ingestion, labeling, features, pipeline
scripts/                raw validation + processed dataset build/validation CLIs
docs/                   data contract, MLOps rollout, deployment/runbooks
notebooks/              EDA and exploratory work only
tests/                  pytest suite
dvc.yaml                reproducible raw-validation + processed-data pipeline
```

## Setup

Requires Python 3.12.3 (chosen for broad wheel/tooling support; 3.14 is too new for some
scientific-stack packages to have published wheels yet).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Raw FEMTO data is versioned with DVC and stored in the team's MinIO bucket. After
configuring the MinIO S3 endpoint and local-only credentials, reproduce the source data with:

```bash
dvc pull
```

The expected source layout is:

```text
data/raw/
├── Training_set/
├── Test_set/
└── Validation_Set/
```

`Training_set` is development/training data, `Test_set` is the official truncated holdout
input, and `Validation_Set` is used only to derive official holdout ground truth. See
`docs/data_contract.md` before changing data/modeling code.

Verify raw data and run tests:

```bash
python scripts/verify_data.py
pytest -q
```

Build the production-safe processed artifacts:

```bash
python scripts/build_datasets.py --mode all
python scripts/verify_processed_data.py
```

Or run the reproducible DVC pipeline:

```bash
dvc repro
dvc push
```

## Local MLOps stack

The development stack includes Airflow, MLflow (including its Model Registry),
Grafana, PostgreSQL, and MinIO-compatible artifact storage. Docker Desktop with
Compose v2 is required.

```bash
cp .env.example .env
docker compose up --build -d
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead of `cp`.

| Service | URL | Local credentials |
| --- | --- | --- |
| Airflow | http://localhost:8080 | `admin` / `admin` by default |
| MLflow + Model Registry | http://localhost:5000 | none (local only) |
| Grafana | http://localhost:3000 | `admin` / `admin` by default |
| MinIO console | http://localhost:9001 | `minio` / `minio-local` by default |
| MinIO S3 API | http://localhost:9000 | DVC / object-storage clients |

In Airflow, enable and manually trigger the `femto_rul_training` DAG. Its smoke-test
model uses synthetic regression data so the integration can be tested before the
FEMTO data pipeline is ready. The run and RMSE appear in MLflow, and the model is
registered as `femto-rul-model`.

Useful commands:

```bash
docker compose ps
docker compose logs -f
docker compose down
docker compose down -v  # also removes local service databases and artifacts
```

The exposed passwords and ports are development defaults only. Replace them with
managed secrets, private networking, TLS, and managed databases/object storage
before deploying this stack to a cloud environment.

## Oracle Cloud

The Ubuntu ARM64 deployment uses the same images with a resource-limited Compose
override, generated secrets, persistent named volumes, and localhost-only web
ports accessed through SSH tunnels. Follow
[docs/oracle_deployment.md](docs/oracle_deployment.md).
