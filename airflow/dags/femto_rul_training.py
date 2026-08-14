"""Starter DAG proving Airflow -> MLflow -> Model Registry integration."""

from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="femto_rul_training",
    description="Train a starter RUL model and register it in MLflow.",
    schedule=None,
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["femto", "rul", "mlflow"],
)
def femto_rul_training():
    @task
    def train_and_register() -> dict[str, float | str]:
        import os

        import mlflow
        import mlflow.sklearn
        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import root_mean_squared_error
        from sklearn.model_selection import train_test_split

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("femto-rul-local")

        features, target = make_regression(
            n_samples=500, n_features=12, noise=15.0, random_state=42
        )
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        parameters = {"n_estimators": 100, "max_depth": 8, "random_state": 42}

        with mlflow.start_run(run_name="airflow-smoke-test") as run:
            model = RandomForestRegressor(**parameters)
            model.fit(x_train, y_train)
            rmse = root_mean_squared_error(y_test, model.predict(x_test))
            mlflow.log_params(parameters)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name="femto-rul-model",
            )
            return {"run_id": run.info.run_id, "rmse": float(rmse)}

    train_and_register()


femto_rul_training()
