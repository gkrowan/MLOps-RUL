"""Builds and saves the Phase 17 Evidently drift/quality report.

Prediction-distribution monitoring: a live prediction is not the same
quantity as train_features.parquet's rul_seconds label (one is a model
output, the other is a ground-truth label), so it is not folded into the
Evidently DataDriftPreset as a formal target/prediction comparison — that
needs production ground truth, which only exists via the isolated holdout
evaluation (Phase 12), not live traffic without labels. Instead,
prediction_sanity_summary() reports plain descriptive statistics for
predicted_rul_seconds next to the training label's descriptive statistics,
as a sanity-range check, not a statistical drift test.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset, DataSummaryPreset

from femto_rul.monitoring.column_mapping import build_data_definition


def build_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Snapshot:
    data_definition = build_data_definition()
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    return report.run(current_data=current_dataset, reference_data=reference_dataset)


def drifted_column_share(snapshot: Snapshot) -> float | None:
    """Fraction of the 24 features flagged as drifted, read from the
    DriftedColumnsCount metric. None if that metric isn't present (would
    mean DataDriftPreset didn't run)."""
    for metric in snapshot.dict()["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            return metric["value"]["share"]
    return None


def prediction_sanity_summary(
    reference_targets: pd.Series, current_predictions: pd.Series
) -> dict:
    """Descriptive-stats comparison — not a drift test, see module docstring."""

    def _describe(series: pd.Series) -> dict:
        series = series.dropna()
        if series.empty:
            return {"count": 0}
        return {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p50": float(series.median()),
            "max": float(series.max()),
        }

    return {
        "training_rul_seconds": _describe(reference_targets),
        "production_predicted_rul_seconds": _describe(current_predictions),
    }


def save_report(
    snapshot: Snapshot, out_dir: Path, prediction_summary: dict | None = None
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot.save_html(str(out_dir / "data_drift.html"))
    snapshot.save_json(str(out_dir / "data_drift.json"))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "drifted_column_share": drifted_column_share(snapshot),
        "prediction_sanity": prediction_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return out_dir
