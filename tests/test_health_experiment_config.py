from femto_rul.experiments.config import HEALTH_DATASET_PATH, load_experiment_config


def test_health_v2_experiments_use_one_fixed_representation():
    cfg = load_experiment_config()
    ids = ["E201", "E202", "E203", "E204"]
    assert all(cfg.model(i).representation == "health_v2" for i in ids)
    assert all(cfg.model(i).benchmark_version == "health-indicator-direct-rul-v2" for i in ids)
    assert HEALTH_DATASET_PATH.name == "prefix_health_v2.parquet"
