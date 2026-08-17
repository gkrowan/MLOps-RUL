"""Evidently schema for Prefix V1 (Phase 17).

evidently==0.7.21 replaced the old ColumnMapping/metric_preset API with
Report + evidently.presets + Dataset.from_pandas(data_definition=...).
This module builds the DataDefinition once from
features.prefix.prefix_feature_columns() — the same source of truth the
served model, the API request schema, and telemetry logging already use —
rather than re-listing the 21 column names a fourth time.
"""

from __future__ import annotations

from evidently import DataDefinition

from femto_rul.features.prefix import prefix_feature_columns


def build_data_definition() -> DataDefinition:
    """DataDefinition for the drift/quality report over the 21 model input
    features. Prediction-distribution monitoring is handled separately in
    report.py (see its docstring for why it isn't folded in here)."""
    return DataDefinition(numerical_columns=list(prefix_feature_columns()))
