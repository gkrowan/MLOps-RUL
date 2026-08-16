"""Tests for src/femto_rul/monitoring/drift_scenarios.py (Phase 18).

Pure synthetic-data tests, no network/DB/Evidently dependency — same
convention as tests/test_monitoring.py.
"""

from femto_rul.features.prefix import prefix_feature_columns
from femto_rul.monitoring.drift_scenarios import (
    amplitude_drift,
    channel_drift,
    extreme_values,
    invalid_schema_requests,
    missing_value_sentinel,
)

PREFIX_FEATURE_COLUMNS = prefix_feature_columns()
CONTEXT_COLUMNS = ["observed_age_seconds", "rotation_speed_rpm", "radial_load_n"]


def _synthetic_row() -> dict:
    return {col: 1.0 for col in PREFIX_FEATURE_COLUMNS}


def _synthetic_reference_ranges() -> dict:
    return {col: {"p01": 0.5, "p99": 2.5} for col in PREFIX_FEATURE_COLUMNS}


def test_amplitude_drift_targets_only_ratio_columns():
    row = _synthetic_row()
    reference_ranges = _synthetic_reference_ranges()

    drifted = amplitude_drift(row, reference_ranges, factor=4.0)

    for col in PREFIX_FEATURE_COLUMNS:
        if col.endswith(("_current_over_early", "_recent_mean_over_early")):
            assert drifted[col] == reference_ranges[col]["p99"] * 4.0
        else:
            assert drifted[col] == row[col]


def test_channel_drift_swaps_all_horiz_vert_pairs_and_nothing_else():
    row = _synthetic_row()
    for col in PREFIX_FEATURE_COLUMNS:
        if "_horiz" in col:
            row[col] = 10.0
        elif "_vert" in col:
            row[col] = 20.0

    drifted = channel_drift(row)

    swapped_pairs = 0
    for col in PREFIX_FEATURE_COLUMNS:
        if "_horiz" in col:
            assert drifted[col] == 20.0
            assert drifted[col.replace("_horiz", "_vert")] == 10.0
            swapped_pairs += 1
    assert swapped_pairs == 9

    for col in CONTEXT_COLUMNS:
        assert drifted[col] == row[col]


def test_missing_value_sentinel_overwrites_only_requested_columns():
    row = _synthetic_row()
    target_columns = ["rms_horiz_current_over_early", "kurtosis_vert_recent_slope_per_hour"]

    drifted = missing_value_sentinel(row, target_columns, sentinel=-999.0)

    for col in target_columns:
        assert drifted[col] == -999.0
    untouched = [c for c in PREFIX_FEATURE_COLUMNS if c not in target_columns]
    for col in untouched:
        assert drifted[col] == row[col]


def test_extreme_values_pushes_requested_columns_past_p99():
    row = _synthetic_row()
    reference_ranges = _synthetic_reference_ranges()
    target_columns = ["radial_load_n", "rms_vert_recent_slope_per_hour"]

    drifted = extreme_values(row, reference_ranges, target_columns, multiplier=10.0)

    for col in target_columns:
        assert drifted[col] == reference_ranges[col]["p99"] * 10.0
    untouched = [c for c in PREFIX_FEATURE_COLUMNS if c not in target_columns]
    for col in untouched:
        assert drifted[col] == row[col]


def test_invalid_schema_requests_are_actually_malformed():
    requests_payloads = invalid_schema_requests()
    assert len(requests_payloads) == 3

    missing_field, extra_field, wrong_type = requests_payloads

    assert set(missing_field.keys()) == set(PREFIX_FEATURE_COLUMNS) - {PREFIX_FEATURE_COLUMNS[0]}

    assert set(extra_field.keys()) == set(PREFIX_FEATURE_COLUMNS) | {"unexpected_field"}

    assert set(wrong_type.keys()) == set(PREFIX_FEATURE_COLUMNS)
    assert isinstance(wrong_type[PREFIX_FEATURE_COLUMNS[0]], str)
