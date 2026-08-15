# Final model release flow

The final development selection is frozen before official holdout access:

- `baseline` — E101 median, best development LOBO RMSE.
- `candidate` — tuned ExtraTrees, strongest learned model after Optuna HPO.

Run:

```bash
python scripts/register_finalists.py
python scripts/evaluate_official_holdout.py
# Use the exact promotion command printed by the evaluation script:
python scripts/promote_champion.py --source-alias baseline   # or candidate
```

`register_finalists.py` logs both models with MLflow and registers them under
`femto-rul-model`. Because the MLflow artifact store is backed by MinIO, the
model package is stored in MinIO automatically; do not manually copy a `.pkl`
file to a bucket.

`evaluate_official_holdout.py` performs one endpoint prediction for each of the
11 truncated Test_set bearings and only then joins against the
Validation_Set-derived ground truth. Treat that evaluation as final: do not
perform further feature engineering or HPO based on its result.
