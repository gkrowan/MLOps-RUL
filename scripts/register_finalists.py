#!/usr/bin/env python3
"""Register the two frozen development finalists in MLflow Model Registry.

The finalists are selected *before* official Test_set/Validation_Set access:
  - baseline alias: E101 median, the development metric champion
  - candidate alias: tuned ExtraTrees, the strongest learned model

Model artifacts are stored by MLflow in its configured artifact store (MinIO in
this project). No manual .pkl upload is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from femto_rul.experiments.config import PREFIX_DATASET_PATH, load_experiment_config
from femto_rul.experiments.models import make_model
from femto_rul.experiments.runner import run_experiment
from femto_rul.experiments.tracking import configure_mlflow, reproducibility_tags
from femto_rul.features.prefix import prefix_feature_columns


def _best_params(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    return dict(payload.get("best_params", payload))


def _registered_version_for_run(client, name: str, run_id: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    matches = [v for v in versions if getattr(v, "run_id", None) == run_id]
    if not matches:
        raise RuntimeError(f"no registered version of {name!r} found for run {run_id}")
    return str(max(matches, key=lambda v: int(v.version)).version)


def _register_one(
    *,
    cfg,
    experiment_id: str,
    alias: str,
    semantic_version: str,
    model_overrides: dict[str, object] | None,
) -> tuple[str, str]:
    spec = cfg.model(experiment_id)
    evaluation = run_experiment(
        experiment_id,
        track_mlflow=False,
        model_overrides=model_overrides,
        config=cfg,
    )

    frame = pd.read_parquet(PREFIX_DATASET_PATH)
    X = frame[prefix_feature_columns()].copy()
    y = frame["rul_seconds"].astype(float)
    model = make_model(
        spec.model_name,
        defaults=cfg.model_defaults,
        random_state=cfg.random_state,
        overrides=model_overrides,
    )
    fitted = clone(model).fit(X, y)

    mlflow = configure_mlflow(cfg.mlflow_experiment_name)
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    registered_name = str(cfg.registry.get("model_name", "femto-rul-model"))
    tags = reproducibility_tags(
        benchmark_version=cfg.benchmark_version,
        experiment_id=f"FINALIST-{experiment_id}",
        model_name=spec.model_name,
        representation="prefix_v1",
    )
    tags.update(
        {
            "selection_status": "frozen_finalist",
            "finalist_alias": alias,
            "official_holdout_accessed": "false",
        }
    )

    with mlflow.start_run(run_name=f"finalist-{alias}-{spec.model_name}", tags=tags) as run:
        mlflow.log_params(
            {
                "source_experiment": experiment_id,
                "finalist_alias": alias,
                "training_rows": len(frame),
                "training_bearings": int(frame["bearing"].nunique()),
                **{f"model__{k}": v for k, v in (model_overrides or {}).items()},
            }
        )
        for key in [
            "mean_rmse",
            "std_rmse",
            "median_rmse",
            "worst_bearing_rmse",
            "mean_mae",
            "mean_r2",
        ]:
            if key in evaluation.summary:
                mlflow.log_metric(f"development__{key}", float(evaluation.summary[key]))

        signature = infer_signature(X, fitted.predict(X))
        mlflow.sklearn.log_model(
            fitted,
            name="model",
            signature=signature,
            input_example=X.head(min(5, len(X))),
            registered_model_name=registered_name,
        )

        client = MlflowClient()
        version = _registered_version_for_run(client, registered_name, run.info.run_id)
        client.set_registered_model_alias(registered_name, alias, version)
        client.set_model_version_tag(registered_name, version, "semantic_version", semantic_version)
        client.set_model_version_tag(registered_name, version, "selection_status", "frozen_finalist")
        client.set_model_version_tag(registered_name, version, "source_experiment", experiment_id)
        return version, run.info.run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extra-trees-params",
        type=Path,
        default=Path("artifacts/modeling/tuning/extra_trees/best_params.json"),
    )
    args = parser.parse_args()
    if not args.extra_trees_params.exists():
        raise SystemExit(f"missing tuned ExtraTrees params: {args.extra_trees_params}")

    cfg = load_experiment_config()
    extra_params = _best_params(args.extra_trees_params)

    baseline_version, _ = _register_one(
        cfg=cfg,
        experiment_id="E101",
        alias="baseline",
        semantic_version="1.0.0-baseline",
        model_overrides=None,
    )
    candidate_version, _ = _register_one(
        cfg=cfg,
        experiment_id="E105",
        alias="candidate",
        semantic_version="1.0.0-rc1",
        model_overrides=extra_params,
    )

    model_name = str(cfg.registry.get("model_name", "femto-rul-model"))
    print("=" * 92)
    print("Frozen finalists registered — official holdout still untouched")
    print("=" * 92)
    print(f"Registered model : {model_name}")
    print(f"baseline alias   : version {baseline_version} (E101 median)")
    print(f"candidate alias  : version {candidate_version} (tuned ExtraTrees)")
    print("Artifacts        : MLflow artifact store / MinIO")
    print("Test/Validation  : NOT ACCESSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
