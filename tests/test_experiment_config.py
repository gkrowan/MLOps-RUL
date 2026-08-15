from pathlib import Path

from femto_rul.experiments.config import load_experiment_config


def test_canonical_config_has_fixed_benchmark(tmp_path: Path):
    path = tmp_path / "params.yaml"
    path.write_text(
        """
experiment:
  mlflow_experiment_name: test
  benchmark_version: prefix-direct-rul-v1
  random_state: 42
  prefix_fractions: [0.40, 0.50, 0.60]
  primary_metric: mean_rmse
  expected_training_bearings: 6
  expected_prefix_rows: 18
  test_accessed: false
  validation_accessed: false
models:
  E104:
    name: random_forest
    description: test
model_defaults:
  random_forest:
    n_estimators: 10
hpo: {}
registry: {}
"""
    )
    cfg = load_experiment_config(path)
    assert cfg.benchmark_version == "prefix-direct-rul-v1"
    assert cfg.prefix_fractions == (0.4, 0.5, 0.6)
    assert cfg.model("E104").model_name == "random_forest"
    assert cfg.test_accessed is False
    assert cfg.validation_accessed is False
