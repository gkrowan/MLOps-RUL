#!/usr/bin/env python3
"""Run leakage-safe Leave-One-Bearing-Out baseline experiments.

This phase intentionally does not use Test_set or Validation_Set. It reads only
``train_features.parquet`` and ``feature_schema.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from femto_rul.config import ARTIFACTS_DIR, FEATURE_SCHEMA_PATH, TRAIN_FEATURES_PATH
from femto_rul.evaluation.cross_validation import (
    leave_one_bearing_out_cv,
    summarize_cv_metrics,
)
from femto_rul.models.baselines import baseline_estimators, load_training_data


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _parse_models(raw: str, available: dict[str, object]) -> list[str]:
    requested = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise SystemExit(
            f"Unknown model(s): {unknown}. Available: {', '.join(available)}"
        )
    if not requested:
        raise SystemExit("At least one model must be selected")
    return requested


def _write_rmse_plot(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("mean_rmse", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ordered["model"], ordered["mean_rmse"], yerr=ordered["std_rmse"])
    ax.set_ylabel("LOBO RMSE (seconds)")
    ax.set_title("FEMTO RUL baseline comparison")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="median,ridge,random_forest,hist_gradient_boosting",
        help="Comma-separated model names",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "modeling" / "baselines_v1",
    )
    args = parser.parse_args()

    X, y, groups, schema = load_training_data(TRAIN_FEATURES_PATH, FEATURE_SCHEMA_PATH)
    estimators = baseline_estimators(random_state=42)
    selected = _parse_models(args.models, estimators)

    print("=" * 72)
    print("FEMTO RUL baseline modeling — Leave-One-Bearing-Out CV")
    print("=" * 72)
    print(f"Training rows: {len(X):,}")
    print(f"Bearings: {groups.nunique()}")
    print(f"Model features: {X.shape[1]}")
    print(f"Feature set: {schema.get('feature_set_version', 'unknown')}")
    print(f"Models: {', '.join(selected)}")
    print("Holdout access: NONE (Training_set only)")
    print()

    all_fold_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []

    for model_name in selected:
        print(f"Running {model_name}...")
        fold_metrics, predictions = leave_one_bearing_out_cv(
            estimators[model_name],
            X,
            y,
            groups,
            model_name=model_name,
            clip_nonnegative=True,
        )
        all_fold_metrics.append(fold_metrics)
        all_predictions.append(predictions)
        mean_rmse = fold_metrics["rmse"].mean()
        std_rmse = fold_metrics["rmse"].std()
        print(f"  mean RMSE: {mean_rmse:,.2f} ± {std_rmse:,.2f} sec")

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_cv_metrics(fold_metrics)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _write_rmse_plot(summary, output_dir / "rmse_comparison.png")

    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_set_version": schema.get("feature_set_version"),
        "cv_strategy": "LeaveOneGroupOut(bearing)",
        "n_folds": int(groups.nunique()),
        "training_rows": int(len(X)),
        "model_feature_count": int(X.shape[1]),
        "models": selected,
        "prediction_constraint": "clip RUL predictions to >= 0 seconds",
        "holdout_used": False,
        "phm12_note": (
            "CV PHM12 value is a snapshot-level diagnostic over positive-RUL rows. "
            "Official challenge-style PHM12 scoring will be computed later from one "
            "endpoint prediction per Test_set bearing."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("Baseline summary (lower RMSE/MAE is better; higher R2/PHM12 is better)")
    print("=" * 72)
    display_cols = [
        "model",
        "mean_rmse",
        "std_rmse",
        "mean_mae",
        "mean_r2",
        "mean_phm12_snapshot_score",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nArtifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
