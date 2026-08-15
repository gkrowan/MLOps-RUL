#!/usr/bin/env python3
"""Promote one already-registered frozen finalist to the MLflow champion alias."""

from __future__ import annotations

import argparse

from femto_rul.experiments.config import load_experiment_config
from femto_rul.experiments.tracking import configure_mlflow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-alias", required=True, choices=["baseline", "candidate"])
    args = parser.parse_args()

    cfg = load_experiment_config()
    mlflow = configure_mlflow(cfg.mlflow_experiment_name)
    from mlflow import MlflowClient

    name = str(cfg.registry.get("model_name", "femto-rul-model"))
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, args.source_alias)
    client.set_registered_model_alias(name, "champion", version.version)
    client.set_model_version_tag(name, version.version, "selection_status", "champion")
    client.set_model_version_tag(name, version.version, "promoted_from_alias", args.source_alias)

    print("=" * 84)
    print("MLflow champion promoted")
    print("=" * 84)
    print(f"Model         : {name}")
    print(f"Version       : {version.version}")
    print(f"Source alias  : {args.source_alias}")
    print("Champion URI  :", f"models:/{name}@champion")
    print("Artifact store: MLflow / MinIO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
