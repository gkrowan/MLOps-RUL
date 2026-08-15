from __future__ import annotations

import pandas as pd

from femto_rul.evaluation.holdout import align_endpoint_features_and_truth, endpoint_ground_truth


def test_endpoint_ground_truth_uses_last_observed_test_snapshot():
    gt = pd.DataFrame(
        {
            "condition": [1, 1, 2, 2],
            "bearing": ["Bearing1_3", "Bearing1_3", "Bearing2_3", "Bearing2_3"],
            "file_index": [1, 2, 1, 3],
            "rul_seconds": [100, 90, 80, 60],
        }
    )
    endpoint = endpoint_ground_truth(gt)
    assert endpoint.set_index("bearing").loc["Bearing1_3", "cut_file_index"] == 2
    assert endpoint.set_index("bearing").loc["Bearing1_3", "rul_seconds"] == 90
    assert endpoint.set_index("bearing").loc["Bearing2_3", "cut_file_index"] == 3


def test_endpoint_alignment_is_one_to_one():
    features = pd.DataFrame(
        {
            "condition": [1, 2],
            "bearing": ["Bearing1_3", "Bearing2_3"],
            "cut_file_index": [2, 3],
            "x": [0.1, 0.2],
        }
    )
    truth = pd.DataFrame(
        {
            "condition": [1, 2],
            "bearing": ["Bearing1_3", "Bearing2_3"],
            "cut_file_index": [2, 3],
            "rul_seconds": [90.0, 60.0],
        }
    )
    joined = align_endpoint_features_and_truth(features, truth)
    assert len(joined) == 2
    assert joined["rul_seconds"].tolist() == [90.0, 60.0]
