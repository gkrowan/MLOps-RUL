from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from femto_rul.experiments.models import make_model


def test_random_forest_factory_applies_overrides():
    model = make_model(
        "random_forest",
        defaults={"random_forest": {"n_estimators": 20, "min_samples_leaf": 2}},
        random_state=7,
        overrides={"n_estimators": 30},
    )
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 30
    assert model.min_samples_leaf == 2
    assert model.random_state == 7


def test_extra_trees_factory():
    model = make_model(
        "extra_trees",
        defaults={"extra_trees": {"n_estimators": 20}},
        random_state=7,
    )
    assert isinstance(model, ExtraTreesRegressor)
    assert model.n_estimators == 20
