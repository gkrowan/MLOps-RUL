#!/usr/bin/env python3
"""Fit the selected canonical model on all six training bearings and register it.

This command does NOT run Test_set or Validation_Set evaluation. Registration is
therefore a development candidate step, not final promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from femto_rul.experiments.config import PREFIX_DATASET_PATH, load_experiment_config
from femto_rul.experiments.models import make_model
from femto_rul.experiments.registry import register_candidate
from femto_rul.experiments.runner import run_experiment
from femto_rul.experiments.tracking import configure_mlflow, reproducibility_tags
from femto_rul.features.prefix import prefix_feature_columns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Canonical experiment ID, e.g. E104")
    parser.add_argument("--best-params", type=Path, default=None, help="Optional Optuna best_params.json")
    parser.add_argument("--semantic-version", default=None)
    parser.add_argument("--alias", default=None)
    args = parser.parse_args()

    cfg = load_experiment_config()
    spec = cfg.model(args.experiment)
    if spec.model_name == "condition_life_prior":
        raise SystemExit("condition_life_prior cannot be registered as the deployable candidate")

    overrides = None
    if args.best_params:
        payload = json.loads(args.best_params.read_text())
        overrides = dict(payload.get("best_params", payload))

    # Re-run the exact LOBO protocol so the registry run carries selection metrics.
    evaluation = run_experiment(
        args.experiment,
        track_mlflow=False,
        model_overrides=overrides,
        config=cfg,
    )

    frame = pd.read_parquet(PREFIX_DATASET_PATH)
    X = frame[prefix_feature_columns()].copy()
    y = frame["rul_seconds"].astype(float)
    model = make_model(
        spec.model_name,
        defaults=cfg.model_defaults,
        random_state=cfg.random_state,
        overrides=overrides,
    )
    fitted = clone(model).fit(X, y)

    mlflow = configure_mlflow(cfg.mlflow_experiment_name)
    import mlflow.sklearn
    from mlflow.models import infer_signature

    registered_name = str(cfg.registry.get("model_name", "femto-rul-model"))
    alias = args.alias or str(cfg.registry.get("candidate_alias", "candidate"))
    semver = args.semantic_version or str(cfg.registry.get("semantic_version", "0.1.0"))
    tags = reproducibility_tags(
        benchmark_version=cfg.benchmark_version,
        experiment_id=f"REGISTER-{args.experiment}",
        model_name=spec.model_name,
    )
    tags["selection_status"] = "candidate"

    with mlflow.start_run(run_name=f"candidate-{args.experiment}-{spec.model_name}", tags=tags) as run:
        for key in ["mean_rmse", "std_rmse", "median_rmse", "worst_bearing_rmse", "mean_mae", "mean_r2"]:
            mlflow.log_metric(key, float(evaluation.summary[key]))
        signature = infer_signature(X, fitted.predict(X))
        mlflow.sklearn.log_model(
            fitted,
            name="model",
            signature=signature,
            input_example=X.head(min(5, len(X))),
            registered_model_name=registered_name,
        )
        version = register_candidate(
            mlflow=mlflow,
            run_id=run.info.run_id,
            registered_model_name=registered_name,
            alias=alias,
            semantic_version=semver,
        )
        print("=" * 84)
        print("Candidate registered")
        print("=" * 84)
        print(f"Model: {registered_name}")
        print(f"Version: {version}")
        print(f"Alias: {alias}")
        print(f"Semantic version tag: {semver}")
        print(f"Source experiment: {args.experiment}")
        print("Test_set / Validation_Set access: NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
