"""Run the Phase 17 Evidently drift/quality report: prefix_train_v1.parquet
(reference) vs. a trailing window of the predictions table (current).

Exits non-zero if the drifted-column share crosses --fail-threshold, so this
can be wired into a scheduled Airflow check later (optional, not required
for Phase 17 acceptance).

Usage: python scripts/run_monitoring_report.py [--window "24 hours"] [--out artifacts/monitoring] [--fail-threshold 0.5]
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from femto_rul import config
from femto_rul.monitoring.current import load_current_features, load_current_predictions
from femto_rul.monitoring.reference import load_reference_features, load_reference_targets
from femto_rul.monitoring.report import (
    build_report,
    drifted_column_share,
    prediction_sanity_summary,
    save_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="24 hours", help="Postgres interval literal")
    parser.add_argument("--out", default=str(config.MONITORING_ARTIFACTS_DIR))
    parser.add_argument("--fail-threshold", type=float, default=0.5)
    args = parser.parse_args()

    print("Loading reference distribution (prefix_train_v1.parquet)...")
    reference_df = load_reference_features()
    reference_targets = load_reference_targets()

    print(f"Loading current distribution (predictions, last {args.window})...")
    current_df = load_current_features(args.window)
    current_predictions = load_current_predictions(args.window)

    if current_df.empty:
        print(f"No successful predictions in the last {args.window}. Nothing to compare.")
        sys.exit(0)

    print(f"Building report: {len(reference_df)} reference rows vs {len(current_df)} current rows")
    snapshot = build_report(reference_df, current_df)
    prediction_summary = prediction_sanity_summary(reference_targets, current_predictions)

    run_dir = Path(args.out) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_report(snapshot, run_dir, prediction_summary)
    print(f"Wrote report to {run_dir}")

    share = drifted_column_share(snapshot)
    print(f"Drifted column share: {share}")

    if share is not None and share > args.fail_threshold:
        print(f"FAIL: drifted column share {share} exceeds threshold {args.fail_threshold}")
        sys.exit(1)


if __name__ == "__main__":
    main()
