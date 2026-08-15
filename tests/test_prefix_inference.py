from __future__ import annotations

import numpy as np
import pandas as pd

from femto_rul.features.prefix import PREFIX_SOURCE_FEATURES, build_prefix_endpoint_features, prefix_feature_columns


def test_build_prefix_endpoint_features_needs_no_rul_labels():
    rows = []
    for bearing, condition in [("Bearing1_3", 1), ("Bearing2_3", 2)]:
        for idx in range(1, 71):
            row = {"condition": condition, "bearing": bearing, "file_index": idx}
            for j, name in enumerate(PREFIX_SOURCE_FEATURES):
                row[name] = float(1.0 + j + 0.01 * idx)
            rows.append(row)
    frame = pd.DataFrame(rows)
    result = build_prefix_endpoint_features(frame)
    assert len(result) == 2
    assert "rul_seconds" not in result.columns
    assert set(prefix_feature_columns()).issubset(result.columns)
    assert np.isfinite(result[prefix_feature_columns()].to_numpy(dtype=float)).all()
    assert set(result["cut_file_index"]) == {70}
