"""Validate local/application configuration and forwarded MLOps services.

This script intentionally reads all hosts/ports from femto_rul.config. It does
not embed cloud addresses or credentials.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from femto_rul import config


SERVICES = {
    "Airflow": config.AIRFLOW_BASE_URL,
    "MLflow": config.MLFLOW_TRACKING_URI,
    "Grafana": config.GRAFANA_BASE_URL,
    "MinIO Console": config.MINIO_CONSOLE_URL,
    "MinIO S3 API": config.MINIO_ENDPOINT_URL,
}


def tcp_check(url: str, timeout: float = 2.0) -> tuple[bool, str]:
    parsed = urlparse(url)
    if not parsed.hostname:
        return False, "invalid URL"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True, f"{parsed.hostname}:{port} reachable"
    except OSError as exc:
        return False, f"{parsed.hostname}:{port} unreachable ({exc})"


def main() -> None:
    print("FEMTO-RUL configuration check")
    print("=" * 72)
    print(f"repo_root              : {config.REPO_ROOT}")
    print(f"data_root              : {config.DATA_ROOT}")
    print(f"raw_data_dir           : {config.RAW_DATA_DIR}")
    print(f"extracted_data_dir     : {config.EXTRACTED_DATA_DIR}")
    print(f"processed_data_dir     : {config.PROCESSED_DATA_DIR}")
    print(f"artifacts_dir          : {config.ARTIFACTS_DIR}")
    print(f"dvc_remote_name        : {config.DVC_REMOTE_NAME}")
    print(f"dvc_remote_url         : {config.DVC_REMOTE_URL or '[not configured]'}")
    print(f"mlflow_experiment_name : {config.MLFLOW_EXPERIMENT_NAME}")
    print(f"mlflow_model_name      : {config.MLFLOW_MODEL_NAME}")
    print()

    failures = 0
    for name, url in SERVICES.items():
        if not url:
            print(f"[CONFIG MISSING] {name:<16} environment variable not set")
            failures += 1
            continue
        ok, detail = tcp_check(url)
        print(f"[{'OK' if ok else 'FAIL':<4}] {name:<16} {url} — {detail}")
        failures += 0 if ok else 1

    print()
    if failures:
        print(f"Environment check completed with {failures} missing/unreachable setting(s).")
        raise SystemExit(1)
    print("Environment check passed.")


if __name__ == "__main__":
    main()
