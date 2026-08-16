"""Train configurable RUL models from PostgreSQL and register them in MLflow."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

from airflow.decorators import dag, get_current_context, task
from airflow.models.param import Param


@dag(
    dag_id="femto_postgres_model_training",
    description="Train and register a configurable model from PostgreSQL features.",
    schedule=None,
    start_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    params={
        "model_type": Param(
            "random_forest",
            type="string",
            enum=["random_forest", "extra_trees", "gradient_boosting"],
        ),
        "hyperparameters": Param({}, type="object"),
        "dataset_version": Param("v1", type="string", minLength=1),
        "feature_table": Param("femto_features", type="string", minLength=1),
        "train_split": Param("Training_set", type="string", minLength=1),
        "evaluation_split": Param("Validation_Set", type="string", minLength=1),
        "experiment_name": Param("femto-rul-postgres", type="string", minLength=1),
        "registered_model_name": Param(
            "femto-rul-model", type="string", minLength=1
        ),
        "run_name": Param("", type=["null", "string"]),
    },
    tags=["femto", "postgres", "mlflow", "training"],
)
def femto_postgres_model_training():
    @task(execution_timeout=timedelta(hours=4))
    def train_and_register() -> dict[str, float | int | str]:
        import json

        import mlflow
        import mlflow.sklearn
        import pandas as pd
        from mlflow.models import infer_signature
        from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
        from sqlalchemy import create_engine, text

        from femto_rul.modeling.training import build_regressor, prepare_feature_splits

        params = get_current_context()["params"]
        table_name = str(params["feature_table"])
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table_name):
            raise ValueError(f"Invalid feature_table: {table_name!r}")

        engine = create_engine(os.environ["FEATURE_STORE_DATABASE_URL"])
        with engine.connect() as connection:
            dataset = pd.read_sql(
                text(f'SELECT * FROM "{table_name}"'), connection
            )
        engine.dispose()
        if dataset.empty:
            raise RuntimeError(f"PostgreSQL table {table_name!r} is empty")

        dataset_version = str(params["dataset_version"])
        if "dataset_version" in dataset.columns:
            dataset = dataset[dataset["dataset_version"] == dataset_version]
            if dataset.empty:
                raise ValueError(
                    f"No rows in {table_name!r} have dataset_version "
                    f"{dataset_version!r}"
                )

        train_split = str(params["train_split"])
        evaluation_split = str(params["evaluation_split"])
        x_train, x_eval, y_train, y_eval, dropped_rows = prepare_feature_splits(
            dataset,
            train_split=train_split,
            evaluation_split=evaluation_split,
        )

        model_type = str(params["model_type"])
        overrides = dict(params["hyperparameters"] or {})
        model, effective_parameters = build_regressor(model_type, overrides)

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(str(params["experiment_name"]))
        run_name = str(params["run_name"] or f"{model_type}-{dataset_version}")

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "model_type": model_type,
                    "dataset_version": dataset_version,
                    "feature_table": table_name,
                    "train_split": train_split,
                    "evaluation_split": evaluation_split,
                }
            )
            mlflow.log_params(
                {
                    "model_type": model_type,
                    "dataset_version": dataset_version,
                    "feature_table": table_name,
                    "train_split": train_split,
                    "evaluation_split": evaluation_split,
                    "train_rows": len(x_train),
                    "evaluation_rows": len(x_eval),
                    "feature_count": len(x_train.columns),
                    "dropped_invalid_rows": dropped_rows,
                    **{
                        f"model__{name}": value
                        for name, value in effective_parameters.items()
                    },
                }
            )

            mlflow_dataset = mlflow.data.from_pandas(
                dataset,
                name=f"{table_name}-{dataset_version}",
                targets="rul_seconds",
            )
            mlflow.log_input(mlflow_dataset, context="training-and-evaluation")

            model.fit(x_train, y_train)
            predictions = model.predict(x_eval)
            metrics = {
                "rmse": float(root_mean_squared_error(y_eval, predictions)),
                "mae": float(mean_absolute_error(y_eval, predictions)),
                "r2": float(r2_score(y_eval, predictions)),
            }
            mlflow.log_metrics(metrics)

            if hasattr(model, "feature_importances_"):
                importance = dict(
                    sorted(
                        zip(x_train.columns, model.feature_importances_),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                mlflow.log_dict(
                    json.loads(json.dumps(importance, default=float)),
                    "feature_importance.json",
                )

            signature = infer_signature(x_train, model.predict(x_train.head(5)))
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=x_train.head(5),
                registered_model_name=str(params["registered_model_name"]),
            )

            return {
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "model_type": model_type,
                "dataset_version": dataset_version,
                **metrics,
            }

    train_and_register()


femto_postgres_model_training()
