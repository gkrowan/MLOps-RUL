import numpy as np
import pandas as pd

from femto_rul.models.prefix_models import ConditionLifePriorRegressor


def test_condition_life_prior_uses_age_plus_rul_training_life():
    X = pd.DataFrame(
        {
            "condition": [1, 1, 2, 2],
            "observed_age_seconds": [100.0, 200.0, 100.0, 200.0],
        }
    )
    y = np.array([900.0, 800.0, 1900.0, 1800.0])
    model = ConditionLifePriorRegressor().fit(X, y)
    pred = model.predict(pd.DataFrame({"condition": [1, 2], "observed_age_seconds": [400.0, 400.0]}))
    assert np.allclose(pred, [600.0, 1600.0])
