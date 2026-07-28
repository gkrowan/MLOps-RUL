# Project Proposal: Predictive Maintenance for Rolling-Element Bearings (FEMTO/PRONOSTIA Dataset)

Dataset available here (#10 in the list): https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Group Members: Anuradha Rakh, Chris Korabik, Erika Yan, Grace Rowan

Dataset: We plan to use the FEMTO Bearing Dataset (PRONOSTIA platform, IEEE PHM 2012 Challenge), which contains run-to-failure vibration data from 17 rolling-element bearings tested under three operating conditions (varying rotational speed and radial load). Each bearing is monitored via two accelerometers sampled at 25.6kHz, with readings captured every 10 seconds until failure. Failure is defined as the point where vibration amplitude exceeds a fixed 20g threshold. The dataset comes with a predefined train/test split: several bearings run to full failure (training), while others are deliberately truncated before failure, with true remaining life held out for evaluation.

Problem statement:  We will frame this as a regression problem predicting Remaining Useful Life (RUL) in seconds/cycles from a bearing's current vibration signal. Because raw waveforms are high-dimensional and non-stationary, our pipeline will extract time-domain (RMS, kurtosis, skewness, crest factor) and frequency-domain (FFT band energy) features per sample before modeling. Our primary evaluation metric will be RMSE on held-out truncated runs, supplemented by the PHM12 Challenge's asymmetric scoring function, which penalizes late predictions more heavily than early ones. This reflects the real-world cost asymmetry of late maintenance/unpredicted failures versus conservative, preventative maintenance. Our choice of model will likely be XGBoost or LightGBM using the extracted features, but we could potentially opt for a transfer learning or fine-tuning approach. 

Architecture/Tools:  For out pipeline we plan to use Airflow to orchestrate feature extraction and training as a DAG, MLflow to track experiments (model type, hyperparameters, RMSE/scoring metrics) and manage a versioned model registry, Docker and FastAPI to containerize the selected model for real-time inference, and EvidentlyAI to monitor incoming feature distributions against a training baseline. For our drift simulation, we can corrupt vibration inputs (e.g., inject out-of-range amplitude spikes or swap accelerometer channels) to verify the monitoring dashboard flags the resulting distributional shift.

See [femto_mlops_project_plan.md](femto_mlops_project_plan.md) for the week-by-week execution plan and team role breakdown.

## Repo structure

```
data/                   raw FEMTO dataset (gitignored — see "Setup" below)
src/femto_rul/          installable package: config, ingestion, labeling, features, splits
scripts/verify_data.py  data integrity verification (delimiter quirks, gaps, shape checks)
docs/data_notes.md      dataset quirks and labeling/schema decisions — read before touching ingestion
notebooks/              EDA and exploratory work
tests/                  pytest suite
```

## Setup

Requires Python 3.12 (chosen for broad wheel/tooling support; 3.14 is too new for some
scientific-stack packages to have published wheels yet).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the FEMTO dataset (link above) and place it under `data/` so it looks like
`data/Training_set/`, `data/Test_set/`, `data/Validation_Set/` (each full of `Bearing{condition}_{unit}/`
directories). `data/` is gitignored — the dataset isn't committed to the repo; put your local
download there, and the team's shared/preprocessed copy will live on the shared drive instead.

Then verify your local copy matches expectations:

```bash
python scripts/verify_data.py
```

Run tests with `pytest`.
