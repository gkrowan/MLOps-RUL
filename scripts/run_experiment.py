#!/usr/bin/env python3
"""Run exactly one canonical model experiment and log it to MLflow."""

from __future__ import annotations

import argparse

from femto_rul.experiments.runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Canonical ID, e.g. E104")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Evaluate locally without writing an MLflow run (useful only for debugging/tests).",
    )
    args = parser.parse_args()

    result = run_experiment(args.experiment, track_mlflow=not args.no_mlflow)
    s = result.summary
    print("=" * 86)
    print(f"{result.experiment_id} — {result.model_name}")
    print("=" * 86)
    print(f"mean RMSE:          {float(s['mean_rmse']):,.2f} sec")
    print(f"std RMSE:           {float(s['std_rmse']):,.2f} sec")
    print(f"median RMSE:        {float(s['median_rmse']):,.2f} sec")
    print(f"worst bearing RMSE:{float(s['worst_bearing_rmse']):>10,.2f} sec")
    print(f"mean MAE:           {float(s['mean_mae']):,.2f} sec")
    print(f"mean R2:            {float(s['mean_r2']):,.4f}")
    print(
        "monotonic violation rate: "
        f"{float(s['mean_monotonic_violation_rate']):.4f}"
    )
    if result.mlflow_run_id:
        print(f"MLflow run: {result.mlflow_run_id}")
    print(f"Artifacts: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
