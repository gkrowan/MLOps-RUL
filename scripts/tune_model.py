#!/usr/bin/env python3
"""Optuna HPO using the same bearing-level LOBO objective for every trial."""

from __future__ import annotations

import argparse
import json

import optuna
import pandas as pd

from femto_rul.config import ARTIFACTS_DIR
from femto_rul.experiments.config import PREFIX_DATASET_PATH, load_experiment_config
from femto_rul.experiments.models import make_model
from femto_rul.experiments.tracking import configure_mlflow, reproducibility_tags
from femto_rul.experiments.tuning import suggest_params
from femto_rul.evaluation.prefix_validation import prefix_lobo_cv
from femto_rul.features.prefix import prefix_feature_columns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["random_forest", "extra_trees", "xgboost", "lightgbm"])
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config()
    if args.model not in cfg.hpo.get("supported_models", []):
        raise SystemExit(f"HPO is not enabled for {args.model}")
    n_trials = args.trials or int(cfg.hpo.get("n_trials", 30))
    timeout = args.timeout or int(cfg.hpo.get("timeout_seconds", 1800))

    frame = pd.read_parquet(PREFIX_DATASET_PATH)
    X = frame[prefix_feature_columns()].copy()
    y = frame["rul_seconds"].astype(float)
    groups = frame["bearing"].astype(str)

    mlflow = None if args.no_mlflow else configure_mlflow(cfg.mlflow_experiment_name)
    parent = None
    if mlflow is not None:
        tags = reproducibility_tags(
            benchmark_version=cfg.benchmark_version,
            experiment_id=f"HPO-{args.model}",
            model_name=args.model,
        )
        parent = mlflow.start_run(run_name=f"HPO-{args.model}", tags=tags)
        mlflow.log_params(
            {
                "model_name": args.model,
                "n_trials": n_trials,
                "timeout_seconds": timeout,
                "cv_strategy": "leave_one_bearing_out",
                "prefix_rows": len(frame),
            }
        )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args.model)
        estimator = make_model(
            args.model,
            defaults=cfg.model_defaults,
            random_state=cfg.random_state,
            overrides=params,
        )
        folds, _ = prefix_lobo_cv(
            estimator,
            X,
            y,
            groups,
            model_name=args.model,
        )
        mean_rmse = float(folds["rmse"].mean())
        worst_rmse = float(folds["rmse"].max())
        trial.set_user_attr("worst_bearing_rmse", worst_rmse)
        if mlflow is not None:
            with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_rmse", mean_rmse)
                mlflow.log_metric("worst_bearing_rmse", worst_rmse)
                for _, row in folds.iterrows():
                    mlflow.log_metric(
                        f"bearing_rmse__{row['held_out_bearing']}", float(row["rmse"])
                    )
        return mean_rmse

    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"femto-rul-{args.model}")
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        out = (ARTIFACTS_DIR / "modeling" / "tuning" / args.model).resolve()
        out.mkdir(parents=True, exist_ok=True)
        trials = study.trials_dataframe()
        trials.to_csv(out / "trials.csv", index=False)
        best = {
            "model_name": args.model,
            "best_value_mean_rmse": float(study.best_value),
            "best_params": study.best_params,
            "benchmark_version": cfg.benchmark_version,
        }
        (out / "best_params.json").write_text(json.dumps(best, indent=2) + "\n")

        print("=" * 84)
        print(f"HPO complete — {args.model}")
        print("=" * 84)
        print(f"Trials completed: {len(study.trials)}")
        print(f"Best mean LOBO RMSE: {study.best_value:,.2f} sec")
        print(json.dumps(study.best_params, indent=2))
        print(f"Artifacts: {out}")

        if mlflow is not None:
            mlflow.log_metric("best_mean_rmse", float(study.best_value))
            mlflow.log_params({f"best__{k}": v for k, v in study.best_params.items()})
            mlflow.log_artifacts(str(out), artifact_path="hpo")
        return 0
    finally:
        if parent is not None:
            mlflow.end_run()


if __name__ == "__main__":
    raise SystemExit(main())
