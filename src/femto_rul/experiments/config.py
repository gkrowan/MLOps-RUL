"""Configuration loader for FEMTO RUL experiment protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from femto_rul.config import REPO_ROOT, TRAIN_FEATURES_PATH

PARAMS_PATH = REPO_ROOT / "params.yaml"
PREFIX_DATASET_PATH = TRAIN_FEATURES_PATH.parent / "prefix_train_v1.parquet"
HEALTH_DATASET_PATH = TRAIN_FEATURES_PATH.parent / "prefix_health_v2.parquet"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    model_name: str
    description: str
    representation: str = "prefix_v1"
    benchmark_version: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    mlflow_experiment_name: str
    benchmark_version: str
    random_state: int
    prefix_fractions: tuple[float, ...]
    primary_metric: str
    expected_training_bearings: int
    expected_prefix_rows: int
    test_accessed: bool
    validation_accessed: bool
    model_defaults: dict[str, dict[str, Any]]
    models: dict[str, ExperimentSpec]
    hpo: dict[str, Any]
    registry: dict[str, Any]

    def model(self, experiment_id: str) -> ExperimentSpec:
        try:
            return self.models[experiment_id]
        except KeyError as exc:
            valid = ", ".join(sorted(self.models))
            raise KeyError(f"unknown experiment {experiment_id!r}; choose one of: {valid}") from exc


def load_raw_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing experiment configuration: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("params.yaml must contain a mapping")
    return data


def load_experiment_config(path: Path = PARAMS_PATH) -> ExperimentConfig:
    raw = load_raw_params(path)
    exp = raw["experiment"]
    model_specs = {
        exp_id: ExperimentSpec(
            experiment_id=exp_id,
            model_name=str(spec["name"]),
            description=str(spec.get("description", "")),
            representation=str(spec.get("representation", "prefix_v1")),
            benchmark_version=(
                str(spec["benchmark_version"])
                if spec.get("benchmark_version") is not None
                else None
            ),
        )
        for exp_id, spec in raw["models"].items()
    }
    fractions = tuple(float(v) for v in exp["prefix_fractions"])
    if not fractions or any(v <= 0.0 or v >= 1.0 for v in fractions):
        raise ValueError("experiment.prefix_fractions must all be between 0 and 1")
    if tuple(sorted(fractions)) != fractions:
        raise ValueError("experiment.prefix_fractions must be sorted ascending")

    return ExperimentConfig(
        mlflow_experiment_name=str(exp["mlflow_experiment_name"]),
        benchmark_version=str(exp["benchmark_version"]),
        random_state=int(exp["random_state"]),
        prefix_fractions=fractions,
        primary_metric=str(exp["primary_metric"]),
        expected_training_bearings=int(exp["expected_training_bearings"]),
        expected_prefix_rows=int(exp["expected_prefix_rows"]),
        test_accessed=bool(exp["test_accessed"]),
        validation_accessed=bool(exp["validation_accessed"]),
        model_defaults={str(name): dict(values or {}) for name, values in raw.get("model_defaults", {}).items()},
        models=model_specs,
        hpo=dict(raw.get("hpo", {})),
        registry=dict(raw.get("registry", {})),
    )
