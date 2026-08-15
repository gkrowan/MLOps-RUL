#!/usr/bin/env python3
"""Run leakage-safe Leave-One-Bearing-Out baseline experiments.

The default model input comes from the current feature schema. For Feature Set
V2 this means the 24 snapshot features plus causal rolling degradation trends.
The official Test/Validation holdout is never accessed.
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
from femto_rul.evaluation.cross_validation import leave_one_bearing_out_cv, summarize_cv_metrics
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
        raise SystemExit(f"Unknown model(s): {unknown}. Available: {', '.join(available)}")
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
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    X, y, groups, schema = load_training_data(TRAIN_FEATURES_PATH, FEATURE_SCHEMA_PATH)
    estimators = baseline_estimators(random_state=42)
    selected = _parse_models(args.models, estimators)
    feature_version = str(schema.get("feature_set_version", "unknown"))
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else ARTIFACTS_DIR / "modeling" / f"baselines_{feature_version}"
    ).resolve()

    print("=" * 72)
    print("FEMTO RUL baseline modeling — Leave-One-Bearing-Out CV")
    print("=" * 72)
    print(f"Training rows: {len(X):,}")
    print(f"Bearings: {groups.nunique()}")
    print(f"Model features: {X.shape[1]}")
    print(f"Feature set: {feature_version}")
    print(f"Models: {', '.join(selected)}")
    print("Holdout access: NONE (Training_set only)")
    print()

    all_fold_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []

    for model_name in selected:
        print(f"Running {model_name}...")
        fold_metrics, predictions = leave_one_bearing_out_cv(
            estimators[model_name], X, y, groups,
            model_name=model_name, clip_nonnegative=True,
        )
        all_fold_metrics.append(fold_metrics)
        all_predictions.append(predictions)
        print(
            f"  mean RMSE: {fold_metrics['rmse'].mean():,.2f} "
            f"± {fold_metrics['rmse'].std():,.2f} sec"
        )

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_cv_metrics(fold_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    _write_rmse_plot(summary, output_dir / "rmse_comparison.png")

    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_set_version": feature_version,
        "cv_strategy": "LeaveOneGroupOut(bearing)",
        "n_folds": int(groups.nunique()),
        "training_rows": int(len(X)),
        "model_feature_count": int(X.shape[1]),
        "models": selected,
        "prediction_constraint": "clip RUL predictions to >= 0 seconds",
        "holdout_used": False,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("Baseline summary (lower RMSE/MAE is better; higher R2 is better)")
    print("=" * 72)
    print(
        summary[["model", "mean_rmse", "std_rmse", "mean_mae", "mean_r2"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nArtifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
