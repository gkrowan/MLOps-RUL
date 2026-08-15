import pandas as pd

from femto_rul.experiments.config import ExperimentConfig, ExperimentSpec
from femto_rul.experiments.runner import run_experiment
from femto_rul.features.prefix import prefix_feature_columns


def _frame() -> pd.DataFrame:
    rows = []
    fractions = (0.4, 0.6, 0.8)
    for bidx in range(3):
        for pidx, fraction in enumerate(fractions):
            row = {
                "condition": bidx + 1,
                "bearing": f"Bearing{bidx+1}_x",
                "cut_fraction": fraction,
                "cut_file_index": 100 + pidx,
                "observed_age_seconds": float((100 + pidx) * 10),
                "rul_seconds": float(3000 - 500 * pidx + 100 * bidx),
            }
            for i, name in enumerate(prefix_feature_columns()):
                row[name] = float(i + pidx + bidx / 10)
            rows.append(row)
    return pd.DataFrame(rows)


def test_runner_uses_same_lobo_protocol_without_mlflow(tmp_path, monkeypatch):
    # Redirect artifact root imported in runner.
    import femto_rul.experiments.runner as runner

    monkeypatch.setattr(runner, "ARTIFACTS_DIR", tmp_path)
    # Keep this unit test independent of optional parquet engines in the test runner.
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: self.to_csv(path, index=index))
    cfg = ExperimentConfig(
        mlflow_experiment_name="test",
        benchmark_version="prefix-direct-rul-v1",
        random_state=42,
        prefix_fractions=(0.4, 0.6, 0.8),
        primary_metric="mean_rmse",
        expected_training_bearings=3,
        expected_prefix_rows=9,
        test_accessed=False,
        validation_accessed=False,
        model_defaults={"random_forest": {"n_estimators": 10, "min_samples_leaf": 1}},
        models={"E104": ExperimentSpec("E104", "random_forest", "test")},
        hpo={},
        registry={},
    )
    result = run_experiment(
        "E104",
        track_mlflow=False,
        config=cfg,
        prefix_frame=_frame(),
    )
    assert len(result.fold_metrics) == 3
    assert set(result.fold_metrics["held_out_bearing"]) == {
        "Bearing1_x",
        "Bearing2_x",
        "Bearing3_x",
    }
    assert result.summary["experiment_id"] == "E104"
    assert result.output_dir.exists()
