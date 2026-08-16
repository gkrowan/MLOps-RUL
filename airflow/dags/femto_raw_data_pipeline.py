"""MinIO raw archives -> verified FEMTO features -> PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.decorators import dag, task


PIPELINE_ROOT = Path("/opt/project/data/airflow_pipeline")
DOWNLOAD_DIR = PIPELINE_ROOT / "downloads"
EXTRACTED_DIR = PIPELINE_ROOT / "extracted"
FEATURES_PATH = Path("/opt/project/data/processed/features.parquet")


@dag(
    dag_id="femto_raw_data_to_postgres",
    description="Download, verify, featurize, and publish the FEMTO dataset.",
    schedule=None,
    start_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["femto", "minio", "postgres", "features"],
)
def femto_raw_data_to_postgres():
    @task(execution_timeout=timedelta(hours=1))
    def download_and_extract() -> str:
        import os

        from femto_rul.orchestration.raw_data import (
            download_raw_archives,
            extract_and_normalize_archives,
        )

        archives = download_raw_archives(
            bucket=os.getenv("RAW_DATA_BUCKET", "raw-data"),
            prefix=os.getenv("RAW_DATA_PREFIX", ""),
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL", "http://minio:9000"),
            destination=DOWNLOAD_DIR,
        )
        return str(extract_and_normalize_archives(archives, EXTRACTED_DIR))

    @task(execution_timeout=timedelta(hours=1))
    def verify_data(data_dir: str) -> str:
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "/opt/project/scripts/verify_data.py",
                "--data-dir",
                data_dir,
            ],
            check=True,
        )
        return data_dir

    @task(execution_timeout=timedelta(hours=4))
    def build_features(data_dir: str) -> str:
        from femto_rul.pipeline import build_full_dataset

        output_path = FEATURES_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features = build_full_dataset(Path(data_dir))
        if features.empty:
            raise RuntimeError("Feature extraction produced an empty dataset")
        features.to_parquet(output_path, index=False)
        return str(output_path)

    @task(execution_timeout=timedelta(hours=1))
    def publish_to_postgres(features_path: str) -> dict[str, int | str]:
        import os
        import re

        import pandas as pd
        from sqlalchemy import create_engine, text

        table_name = os.getenv("FEATURE_TABLE_NAME", "femto_features")
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table_name):
            raise ValueError(f"Invalid FEATURE_TABLE_NAME: {table_name!r}")

        database_url = os.environ["FEATURE_STORE_DATABASE_URL"]
        staging_table = f"{table_name}__staging"
        features = pd.read_parquet(features_path)
        if features.empty:
            raise RuntimeError("Refusing to publish an empty feature dataset")

        engine = create_engine(database_url)
        features.to_sql(
            staging_table,
            engine,
            if_exists="replace",
            index=False,
            chunksize=1_000,
            method="multi",
        )
        with engine.begin() as connection:
            connection.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
            connection.execute(
                text(f'ALTER TABLE "{staging_table}" RENAME TO "{table_name}"')
            )
            connection.execute(
                text(
                    f'CREATE UNIQUE INDEX "{table_name}_snapshot_uidx" '
                    f'ON "{table_name}" (split, bearing, file_index)'
                )
            )
        engine.dispose()
        return {
            "table": table_name,
            "rows": len(features),
            "columns": len(features.columns),
        }

    extracted = download_and_extract()
    verified = verify_data(extracted)
    features = build_features(verified)
    publish_to_postgres(features)


femto_raw_data_to_postgres()
