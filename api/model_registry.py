"""Small, testable helpers for selecting a registered MLflow model version."""

from __future__ import annotations

from typing import Any


def resolve_model_version(client: Any, model_name: str, reference: str) -> Any:
    """Resolve an MLflow alias or the numerically latest registered version."""
    if reference.lower() != "latest":
        return client.get_model_version_by_alias(model_name, reference)

    versions = list(client.search_model_versions(f"name='{model_name}'"))
    if not versions:
        raise RuntimeError(f"No registered versions exist for model {model_name!r}")
    return max(versions, key=lambda version: int(version.version))
