"""Run validation-set inference through the deployed FastAPI service.

The DAG deliberately does not write to the predictions table. Each request is
sent through ``POST /predict``, whose existing telemetry path records the
request and prediction in PostgreSQL exactly like normal API traffic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.python import get_current_context


@dag(
    dag_id="femto_validation_api_inference",
    description="Build validation prefixes and send them through the prediction API.",
    schedule="@hourly",
    start_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    # Avoid automatically duplicating already-logged API calls after a partial
    # failure. A failed run can be inspected and retriggered intentionally.
    default_args={"retries": 0},
    params={
        "feature_table": Param("femto_features", type="string", minLength=1),
        "validation_split": Param("Validation_Set", type="string", minLength=1),
        "dataset_version": Param("", type=["null", "string"]),
        "api_url": Param("http://api:8000", type="string", minLength=1),
        "request_timeout_seconds": Param(30, type="integer", minimum=1, maximum=300),
    },
    tags=["femto", "validation", "api", "inference"],
)
def femto_validation_api_inference():
    @task(execution_timeout=timedelta(hours=1))
    def run_inference() -> dict[str, int | str]:
        import json
        import os
        import re
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        import pandas as pd
        from sqlalchemy import create_engine, text

        from femto_rul.features.prefix import (
            build_prefix_endpoint_features,
            prefix_feature_columns,
        )

        params = get_current_context()["params"]
        table_name = str(params["feature_table"])
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table_name):
            raise ValueError(f"Invalid feature_table: {table_name!r}")

        split = str(params["validation_split"])
        dataset_version = str(params.get("dataset_version") or "")
        api_url = str(params["api_url"]).rstrip("/")
        timeout = int(params["request_timeout_seconds"])

        query = f'SELECT * FROM "{table_name}" WHERE split = :split'
        query_params: dict[str, str] = {"split": split}
        if dataset_version:
            query += " AND dataset_version = :dataset_version"
            query_params["dataset_version"] = dataset_version
        query += " ORDER BY bearing, file_index"

        engine = create_engine(os.environ["FEATURE_STORE_DATABASE_URL"])
        try:
            with engine.connect() as connection:
                validation = pd.read_sql(text(query), connection, params=query_params)
        finally:
            engine.dispose()

        if validation.empty:
            version_message = (
                f" and dataset_version {dataset_version!r}" if dataset_version else ""
            )
            raise ValueError(
                f"No rows in {table_name!r} have split {split!r}{version_message}"
            )

        # The endpoint builder intentionally rejects labels. They are not part
        # of an inference request and must never be sent to the API.
        endpoint_input = validation.drop(columns=["rul_seconds"], errors="ignore")
        endpoints = build_prefix_endpoint_features(endpoint_input)
        feature_columns = prefix_feature_columns()

        succeeded = 0
        failures: list[str] = []
        for row in endpoints.itertuples(index=False):
            payload = {
                name: float(getattr(row, name))
                for name in feature_columns
            }
            request = Request(
                f"{api_url}/predict",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            bearing = str(getattr(row, "bearing"))
            try:
                with urlopen(request, timeout=timeout) as response:
                    if response.status != 200:
                        failures.append(f"{bearing}: HTTP {response.status}")
                    else:
                        succeeded += 1
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                failures.append(f"{bearing}: HTTP {exc.code}: {body[:300]}")
            except URLError as exc:
                failures.append(f"{bearing}: {exc.reason}")

        if failures:
            sample = "; ".join(failures[:5])
            raise RuntimeError(
                f"API inference failed for {len(failures)} of {len(endpoints)} "
                f"validation bearings. First failures: {sample}"
            )

        return {
            "feature_table": table_name,
            "validation_split": split,
            "dataset_version": dataset_version or "unfiltered",
            "source_rows": len(validation),
            "predictions_logged_by_api": succeeded,
        }

    run_inference()


femto_validation_api_inference()
