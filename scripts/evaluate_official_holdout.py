#!/usr/bin/env python3
"""Run the single official endpoint holdout validation for frozen finalists.

This is the intentional boundary crossing where Test_set features and the
Validation_Set-derived ground truth are brought together. Do not use this
script for further model tuning after viewing the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from femto_rul.config import ARTIFACTS_DIR, TEST_FEATURES_PATH, TEST_GROUND_TRUTH_PATH
from femto_rul.evaluation.holdout import (
    align_endpoint_features_and_truth,
    endpoint_ground_truth,
    official_endpoint_metrics,
)
from femto_rul.experiments.config import load_experiment_config
from femto_rul.experiments.tracking import configure_mlflow, reproducibility_tags
from femto_rul.features.prefix import build_prefix_endpoint_features, prefix_feature_columns


def main() -> int:
    cfg = load_experiment_config()
    model_name = str(cfg.registry.get("model_name", "femto-rul-model"))

    test = pd.read_parquet(TEST_FEATURES_PATH)
    if "rul_seconds" in test.columns:
        raise SystemExit("ABORT: Test_set inference features unexpectedly contain rul_seconds")
    raw_truth = pd.read_parquet(TEST_GROUND_TRUTH_PATH)

    endpoint_features = build_prefix_endpoint_features(test)
    endpoint_truth = endpoint_ground_truth(raw_truth)
    evaluation = align_endpoint_features_and_truth(endpoint_features, endpoint_truth)
    if len(evaluation) != 11:
        raise SystemExit(f"ABORT: expected 11 official Test bearings, found {len(evaluation)}")

    mlflow = configure_mlflow(cfg.mlflow_experiment_name)
    baseline = mlflow.pyfunc.load_model(f"models:/{model_name}@baseline")
    candidate = mlflow.pyfunc.load_model(f"models:/{model_name}@candidate")

    X = evaluation[prefix_feature_columns()].copy()
    evaluation["baseline_prediction_rul_seconds"] = baseline.predict(X)
    evaluation["candidate_prediction_rul_seconds"] = candidate.predict(X)

    actual = evaluation["rul_seconds"].astype(float)
    baseline_metrics = official_endpoint_metrics(actual, evaluation["baseline_prediction_rul_seconds"])
    candidate_metrics = official_endpoint_metrics(actual, evaluation["candidate_prediction_rul_seconds"])

    out = (ARTIFACTS_DIR / "modeling" / "official_holdout").resolve()
    out.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(out / "official_endpoint_predictions.csv", index=False)
    summary = pd.DataFrame(
        [
            {"alias": "baseline", "model": "median", **baseline_metrics},
            {"alias": "candidate", "model": "extra_trees_tuned", **candidate_metrics},
        ]
    ).sort_values("rmse")
    summary.to_csv(out / "official_endpoint_summary.csv", index=False)

    winner = str(summary.iloc[0]["alias"])
    manifest = {
        "scope": "official_test_endpoint_validation",
        "test_bearings": int(len(evaluation)),
        "registered_model": model_name,
        "baseline_uri": f"models:/{model_name}@baseline",
        "candidate_uri": f"models:/{model_name}@candidate",
        "winner_by_rmse": winner,
        "warning": "Official holdout has now been accessed. Do not tune models on these results.",
    }
    (out / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tags = reproducibility_tags(
        benchmark_version="official-holdout-v1",
        experiment_id="FINAL-HOLDOUT",
        model_name="frozen_finalists",
        representation="prefix_v1",
    )
    tags.update(
        {
            "test_accessed": "true",
            "validation_accessed": "true",
            "evaluation_scope": "official_test_endpoint",
            "selection_frozen_before_holdout": "true",
            "no_further_tuning": "true",
        }
    )
    with mlflow.start_run(run_name="FINAL-OFFICIAL-HOLDOUT", tags=tags):
        for prefix, metrics in [("baseline", baseline_metrics), ("candidate", candidate_metrics)]:
            for metric, value in metrics.items():
                mlflow.log_metric(f"{prefix}__{metric}", float(value))
        mlflow.log_param("test_bearings", len(evaluation))
        mlflow.log_param("winner_by_rmse", winner)
        mlflow.log_artifacts(str(out), artifact_path="official_holdout")

    print("=" * 102)
    print("FINAL OFFICIAL HOLDOUT — one endpoint prediction per Test_set bearing")
    print("=" * 102)
    print(summary.round(3).to_string(index=False))
    print("\nWinner by official RMSE:", winner)
    print("\nIMPORTANT: Validation_Set-derived ground truth has now been accessed.")
    print("Do not tune or redesign models using this result.")
    print(f"Artifacts: {out}")
    print("\nPromotion command:")
    print(f"  python scripts/promote_champion.py --source-alias {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
