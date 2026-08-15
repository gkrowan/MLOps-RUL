"""Evidently schema for Feature Set V1 (Phase 17).

evidently==0.7.21 replaced the old ColumnMapping/metric_preset API with
Report + evidently.presets + Dataset.from_pandas(data_definition=...).
This module builds the DataDefinition once from
pipeline.FEATURE_COLUMNS_V1 — the same source of truth the feature
extraction and telemetry logging already use — rather than re-listing the
24 column names a third time.
"""

from __future__ import annotations

from evidently import DataDefinition

from femto_rul.pipeline import FEATURE_COLUMNS_V1


def build_data_definition() -> DataDefinition:
    """DataDefinition for the drift/quality report over the 24 model input
    features. Prediction-distribution monitoring is handled separately in
    report.py (see its docstring for why it isn't folded in here)."""
    return DataDefinition(numerical_columns=list(FEATURE_COLUMNS_V1))
