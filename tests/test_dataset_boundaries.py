from __future__ import annotations

from pathlib import Path

import numpy as np

from femto_rul.pipeline import (
    build_test_feature_dataset,
    build_test_ground_truth,
    build_training_dataset,
)


def _write_acc(path: Path, index: int, delimiter: str = ",") -> None:
    path.mkdir(parents=True, exist_ok=True)
    n = 64
    t = np.arange(n)
    # Six raw columns expected by the production loader.
    data = np.column_stack(
        [
            np.zeros(n),
            np.zeros(n),
            np.full(n, index),
            t,
            np.sin(2 * np.pi * t / 16) + index * 0.01,
            np.cos(2 * np.pi * t / 13) + index * 0.02,
        ]
    )
    np.savetxt(
        path / f"acc_{index:05d}.csv",
        data,
        delimiter=delimiter,
        fmt="%.8f",
    )


def _make_bearing(root: Path, name: str, snapshots: int, delimiter: str = ",") -> Path:
    bearing = root / name
    for idx in range(1, snapshots + 1):
        _write_acc(bearing, idx, delimiter=delimiter)
    return bearing


def test_production_dataset_boundaries(tmp_path: Path) -> None:
    training_dir = tmp_path / "Training_set"
    test_dir = tmp_path / "Test_set"
    validation_dir = tmp_path / "Validation_Set"

    _make_bearing(training_dir, "Bearing1_1", snapshots=3)
    _make_bearing(test_dir, "Bearing1_3", snapshots=2)
    _make_bearing(validation_dir, "Bearing1_3", snapshots=4)

    train = build_training_dataset(training_dir)
    test_features = build_test_feature_dataset(test_dir)
    truth = build_test_ground_truth(test_dir, validation_dir)

    assert "rul_seconds" in train.columns
    assert "rul_seconds" not in test_features.columns
    assert "rul_seconds" in truth.columns

    assert train["rul_seconds"].tolist() == [20, 10, 0]
    assert truth["rul_seconds"].tolist() == [30, 20]

    test_keys = set(
        test_features[["condition", "bearing", "file_index"]]
        .itertuples(index=False, name=None)
    )
    truth_keys = set(
        truth[["condition", "bearing", "file_index"]]
        .itertuples(index=False, name=None)
    )
    assert test_keys == truth_keys


def test_test_feature_builder_never_requires_validation_set(tmp_path: Path) -> None:
    test_dir = tmp_path / "Test_set"
    _make_bearing(test_dir, "Bearing2_7", snapshots=2, delimiter=";")

    # No Validation_Set directory is created. This is a hard production contract.
    test_features = build_test_feature_dataset(test_dir)

    assert len(test_features) == 2
    assert "rul_seconds" not in test_features.columns
    assert test_features["split"].unique().tolist() == ["Test_set"]
