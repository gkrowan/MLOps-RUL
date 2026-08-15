"""MLflow helpers and reproducibility metadata."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

from femto_rul.config import MLFLOW_TRACKING_URI, REPO_ROOT, env

MLFLOW_RUN_OWNER = env("MLFLOW_RUN_OWNER", "unknown") or "unknown"
MLFLOW_TEAM = env("MLFLOW_TEAM", "FEMTO-RUL") or "FEMTO-RUL"


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def reproducibility_tags(
    *,
    benchmark_version: str,
    experiment_id: str,
    model_name: str,
    representation: str = "prefix_v1",
) -> dict[str, str]:
    return {
        "project_contributor": MLFLOW_RUN_OWNER,
        "team": MLFLOW_TEAM,
        "experiment_id": experiment_id,
        "benchmark_version": benchmark_version,
        "model_name": model_name,
        "representation": representation,
        "target_formulation": "direct_rul",
        "cv_strategy": "leave_one_bearing_out",
        "test_accessed": "false",
        "validation_accessed": "false",
        "git_sha": git_sha(),
        "dvc_lock_sha256": sha256_file(REPO_ROOT / "dvc.lock"),
        "params_sha256": sha256_file(REPO_ROOT / "params.yaml"),
        "python_version": platform.python_version(),
    }


def configure_mlflow(experiment_name: str) -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed. Run: pip install -r requirements-modeling.txt"
        ) from exc
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    return mlflow
