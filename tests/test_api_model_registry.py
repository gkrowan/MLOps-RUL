from types import SimpleNamespace

import pytest

from api.model_registry import resolve_model_version


def test_resolve_latest_model_version_numerically() -> None:
    client = SimpleNamespace(
        search_model_versions=lambda _: [
            SimpleNamespace(version="2"),
            SimpleNamespace(version="10"),
        ]
    )

    assert resolve_model_version(client, "model", "latest").version == "10"


def test_resolve_alias_uses_registry_alias() -> None:
    expected = SimpleNamespace(version="4")
    client = SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: expected,
    )

    assert resolve_model_version(client, "model", "champion") is expected


def test_resolve_latest_requires_a_registered_version() -> None:
    client = SimpleNamespace(search_model_versions=lambda _: [])

    with pytest.raises(RuntimeError, match="No registered versions"):
        resolve_model_version(client, "model", "latest")
