import optuna

from femto_rul.experiments.tuning import suggest_params


def test_random_forest_hpo_space_produces_valid_params():
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        params = suggest_params(trial, "random_forest")
        assert params["n_estimators"] >= 300
        assert params["min_samples_leaf"] >= 1
        return 1.0

    study.optimize(objective, n_trials=1)
    assert len(study.trials) == 1
