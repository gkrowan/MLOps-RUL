"""Builds abnormal-but-realistic /predict request payloads for Phase 18
drift simulation (see docs/phase_18_drift_simulation.md).

Pure functions only — no network/DB/Evidently dependency, so these are
unit-testable the same way as column_mapping.py/report.py (Phase 17).
scripts/simulate_drift.py is the only caller that sends these over HTTP.
"""

from __future__ import annotations

from femto_rul.features.prefix import prefix_feature_columns

# The ratio-type columns amplitude drift targets: rising vibration energy
# shows up as these ratios departing from ~1.0, not in the slope columns.
_RATIO_SUFFIXES = ("_current_over_early", "_recent_mean_over_early")


def amplitude_drift(row: dict, reference_ranges: dict, factor: float = 4.0) -> dict:
    """Simulates rising vibration energy: every ratio-type column
    (current_over_early / recent_mean_over_early, 12 of 21 columns) is set
    to `factor`x that column's reference p99."""
    result = dict(row)
    for col in row:
        if col.endswith(_RATIO_SUFFIXES):
            result[col] = reference_ranges[col]["p99"] * factor
    return result


def channel_drift(row: dict) -> dict:
    """Simulates a horiz/vert sensor or wiring swap: every _horiz/_vert
    column pair (9 pairs, all 18 derived feature columns) is swapped.
    The 3 context columns (observed_age_seconds, rotation_speed_rpm,
    radial_load_n) aren't per-channel, so they're untouched."""
    result = dict(row)
    for col in row:
        if "_horiz" in col:
            vert_col = col.replace("_horiz", "_vert")
            if vert_col in row:
                result[col], result[vert_col] = row[vert_col], row[col]
    return result


def missing_value_sentinel(
    row: dict, columns: list[str], sentinel: float = -999.0
) -> dict:
    """A live model can't receive a true null for a required field, so a
    sentinel value is the realistic stand-in for "missing" data."""
    result = dict(row)
    for col in columns:
        result[col] = sentinel
    return result


def extreme_values(
    row: dict, reference_ranges: dict, columns: list[str], multiplier: float = 10.0
) -> dict:
    """Pushes the given columns to `multiplier`x beyond their reference
    p99 — generic out-of-range values, unlike amplitude_drift's fixed
    domain-specific ratio-column set."""
    result = dict(row)
    for col in columns:
        result[col] = reference_ranges[col]["p99"] * multiplier
    return result


def invalid_schema_requests() -> list[dict]:
    """Deliberately malformed raw request bodies: missing a required
    field, an extra/misspelled field (BearingFeatures is extra="forbid"),
    and a wrong-type value. Expected to 422 at the live endpoint before
    ever reaching log_prediction() — see docs/phase_18_drift_simulation.md
    §2."""
    columns = prefix_feature_columns()
    base = {name: 1.0 for name in columns}

    missing_field = dict(base)
    del missing_field[columns[0]]

    extra_field = dict(base)
    extra_field["unexpected_field"] = 1.0

    wrong_type = dict(base)
    wrong_type[columns[0]] = "not-a-number"

    return [missing_field, extra_field, wrong_type]
