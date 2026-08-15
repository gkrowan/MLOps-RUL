# Experiment Commands

```bash
# Modeling dependencies
pip install -r requirements-modeling.txt

# Make sure MLflow + MinIO tunnel is running
curl -s -o /dev/null -w "MLflow: %{http_code}\n" http://127.0.0.1:5000/health
curl -s -o /dev/null -w "MinIO: %{http_code}\n" http://127.0.0.1:9000/minio/health/live

# Rebuild/version canonical prefix data
python -m pytest -q
dvc repro
python scripts/verify_prefix_dataset.py
dvc push

# One experiment
python scripts/run_experiment.py --experiment E104

# Full fixed comparison suite
python scripts/run_experiment_suite.py

# HPO only after the fixed suite identifies competitive families
python scripts/tune_model.py --model random_forest --trials 30
python scripts/tune_model.py --model xgboost --trials 30
python scripts/tune_model.py --model lightgbm --trials 30

# Register a chosen candidate after reviewing MLflow
python scripts/train_candidate.py --experiment E104 --semantic-version 0.1.0

# If using tuned params
python scripts/train_candidate.py \
  --experiment E107 \
  --best-params artifacts/modeling/tuning/xgboost/best_params.json \
  --semantic-version 0.2.0
```

Open MLflow at `http://localhost:5000` and compare runs in the `femto-rul-development` experiment.
