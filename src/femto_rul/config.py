"""Central configuration for the FEMTO/PRONOSTIA MLOps project.

Configuration policy
--------------------
* Dataset/scientific constants live in source control because they define the
  ML problem and must remain identical across environments.
* Environment-specific paths, service URLs, bucket/model names, and runtime
  settings come from environment variables.
* A local .env file is supported for developer convenience.
* Environment variables supplied by Docker, Airflow, CI, or cloud
  infrastructure take precedence over .env values.
* Secrets are never committed to Git and are not exposed as module-level
  constants unless an application component explicitly requires them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


# ---------------------------------------------------------------------------
# Repository root and local .env loading
# ---------------------------------------------------------------------------

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# Local developer convenience only.
# Existing shell / Docker / Airflow / CI environment variables win because
# override=False.
if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_from_env(name: str, default: Path) -> Path:
    """Return a normalized absolute path from env or a repository default.

    Relative paths are resolved from REPO_ROOT so behavior stays consistent
    when code is called from notebooks, scripts, Airflow, tests, or FastAPI.
    """
    raw = os.getenv(name)

    if not raw:
        return default.resolve()

    path = Path(raw).expanduser()

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path.resolve()


def env(name: str, default: str | None = None) -> str | None:
    """Read a non-secret environment setting."""
    return os.getenv(name, default)


def require_env(name: str) -> str:
    """Return a required environment variable or fail clearly.

    Use this for secrets only at the point where they are actually needed,
    for example DVC/MinIO setup.
    """
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            "For local development copy .env.example to .env and set the "
            "required value. For Docker/Airflow/cloud configure it in the "
            "runtime environment."
        )

    return value


# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------

APP_ENV: Final[str] = env("APP_ENV", "local") or "local"


# ---------------------------------------------------------------------------
# Repository / data paths
# ---------------------------------------------------------------------------

DATA_ROOT: Final[Path] = _path_from_env(
    "FEMTO_DATA_ROOT",
    REPO_ROOT / "data",
)

# Backward-compatible alias for older modules/scripts.
DATA_DIR: Final[Path] = DATA_ROOT

# Original FEMTO source data.
#
# Current repository layout:
#
# data/raw/
# ├── Training_set/
# ├── Test_set/
# └── Validation_Set/
#
# DVC will version these source directories directly.
RAW_DATA_DIR: Final[Path] = _path_from_env(
    "FEMTO_RAW_DATA_DIR",
    DATA_ROOT / "raw",
)

# Parent directory containing the released FEMTO split directories.
# Kept separately so the physical storage layout can change later without
# changing ingestion code. Currently this is the same as RAW_DATA_DIR.
EXTRACTED_DATA_DIR: Final[Path] = _path_from_env(
    "FEMTO_EXTRACTED_DATA_DIR",
    RAW_DATA_DIR,
)

# Generated ML-ready datasets.
PROCESSED_DATA_DIR: Final[Path] = _path_from_env(
    "FEMTO_PROCESSED_DATA_DIR",
    DATA_ROOT / "processed",
)

# Generated plots, reports, evaluation output and monitoring artifacts.
ARTIFACTS_DIR: Final[Path] = _path_from_env(
    "FEMTO_ARTIFACTS_DIR",
    REPO_ROOT / "artifacts",
)


# ---------------------------------------------------------------------------
# Dataset split locations
# ---------------------------------------------------------------------------

TRAINING_SET_DIR: Final[Path] = EXTRACTED_DATA_DIR / "Training_set"
TEST_SET_DIR: Final[Path] = EXTRACTED_DATA_DIR / "Test_set"
VALIDATION_SET_DIR: Final[Path] = EXTRACTED_DATA_DIR / "Validation_Set"


# ---------------------------------------------------------------------------
# Processed dataset locations
# ---------------------------------------------------------------------------

TRAIN_FEATURES_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "train_features.parquet"
)

TEST_FEATURES_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "test_features.parquet"
)

TEST_GROUND_TRUTH_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "test_ground_truth.parquet"
)

FEATURE_SCHEMA_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "feature_schema.json"
)


# ---------------------------------------------------------------------------
# External MLOps services
#
# No cloud hostname, localhost port, bucket name, or model name should be
# hard-coded inside training, API, Airflow, DVC, or monitoring modules.
# ---------------------------------------------------------------------------

AIRFLOW_BASE_URL: Final[str | None] = env("AIRFLOW_BASE_URL")
MLFLOW_TRACKING_URI: Final[str | None] = env("MLFLOW_TRACKING_URI")
GRAFANA_BASE_URL: Final[str | None] = env("GRAFANA_BASE_URL")

# MinIO S3-compatible API used by DVC and object-storage clients.
MINIO_ENDPOINT_URL: Final[str | None] = env("MINIO_ENDPOINT_URL")

# Human-facing MinIO browser console.
MINIO_CONSOLE_URL: Final[str | None] = env("MINIO_CONSOLE_URL")


# ---------------------------------------------------------------------------
# DVC / object storage
# ---------------------------------------------------------------------------

DVC_REMOTE_NAME: Final[str] = (
    env("DVC_REMOTE_NAME", "minio") or "minio"
)

DVC_REMOTE_URL: Final[str | None] = env("DVC_REMOTE_URL")


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

MLFLOW_ARTIFACT_BUCKET: Final[str] = (
    env("MLFLOW_ARTIFACT_BUCKET", "mlflow") or "mlflow"
)

MLFLOW_EXPERIMENT_NAME: Final[str] = (
    env("MLFLOW_EXPERIMENT_NAME", "femto-rul") or "femto-rul"
)

MLFLOW_MODEL_NAME: Final[str] = (
    env("MLFLOW_MODEL_NAME", "femto-rul-model") or "femto-rul-model"
)

MLFLOW_MODEL_ALIAS: Final[str] = (
    env("MLFLOW_MODEL_ALIAS", "candidate") or "candidate"
)


# ---------------------------------------------------------------------------
# Dataset / scientific constants
#
# These intentionally remain version-controlled because they define the
# problem rather than the deployment environment.
# ---------------------------------------------------------------------------

ACCELEROMETER_SAMPLING_RATE_HZ: Final[int] = 25_600

# Each vibration CSV contains a 0.1-second snapshot:
# 25,600 samples/sec × 0.1 sec = 2,560 samples.
ACC_SAMPLES_PER_FILE: Final[int] = 2_560

ACC_COLUMNS: Final[list[str]] = [
    "hour",
    "minute",
    "second",
    "microsecond",
    "horiz_accel_g",
    "vert_accel_g",
]

TEMP_SAMPLING_RATE_HZ: Final[int] = 10
TEMP_SAMPLES_PER_FILE: Final[int] = 600

TEMP_COLUMNS: Final[list[str]] = [
    "hour",
    "minute",
    "second",
    "hundredth_second",
    "temperature_c",
]

# New vibration snapshots are recorded approximately every 10 seconds.
FILE_INTERVAL_SECONDS: Final[int] = 10

# operating condition -> machine settings, per FEMTO / PHM12
CONDITIONS: Final[dict[int, dict[str, int]]] = {
    1: {
        "rotation_speed_rpm": 1800,
        "radial_load_n": 4000,
    },
    2: {
        "rotation_speed_rpm": 1650,
        "radial_load_n": 4200,
    },
    3: {
        "rotation_speed_rpm": 1500,
        "radial_load_n": 5000,
    },
}

MLFLOW_RUN_OWNER = env("MLFLOW_RUN_OWNER", "unknown")
MLFLOW_TEAM = env("MLFLOW_TEAM", "FEMTO-RUL")