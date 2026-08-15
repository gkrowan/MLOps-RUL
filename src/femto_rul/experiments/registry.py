"""MLflow Model Registry helper for a deliberately selected candidate model."""

from __future__ import annotations

from typing import Any


def register_candidate(
    *,
    mlflow: Any,
    run_id: str,
    registered_model_name: str,
    alias: str,
    semantic_version: str,
) -> str:
    from mlflow import MlflowClient

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    candidates = [v for v in versions if getattr(v, "run_id", None) == run_id]
    if not candidates:
        raise RuntimeError(
            f"no registered model version for {registered_model_name!r} was created by run {run_id}"
        )
    version = max(candidates, key=lambda v: int(v.version))
    client.set_registered_model_alias(registered_model_name, alias, version.version)
    client.set_model_version_tag(
        registered_model_name, version.version, "semantic_version", semantic_version
    )
    client.set_model_version_tag(
        registered_model_name, version.version, "selection_status", "candidate"
    )
    return str(version.version)
