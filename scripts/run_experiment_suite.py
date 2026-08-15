#!/usr/bin/env python3
"""Run the fixed canonical baseline/model-family comparison suite."""

from __future__ import annotations

import argparse

import pandas as pd

from femto_rul.config import ARTIFACTS_DIR
from femto_rul.experiments.config import load_experiment_config
from femto_rul.experiments.models import MissingModelDependency
from femto_rul.experiments.runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment IDs. Default: all canonical E100-E108 runs.",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config()
    ids = args.experiments or sorted(cfg.models)
    rows: list[dict[str, object]] = []

    for experiment_id in ids:
        spec = cfg.model(experiment_id)
        print(f"\n>>> {experiment_id}: {spec.model_name}")
        try:
            result = run_experiment(
                experiment_id,
                track_mlflow=not args.no_mlflow,
                config=cfg,
            )
        except MissingModelDependency as exc:
            print(f"SKIPPED: {exc}")
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "model": spec.model_name,
                    "status": "skipped_missing_dependency",
                }
            )
            continue
        row = {"experiment_id": experiment_id, "model": spec.model_name, "status": "ok"}
        row.update(result.summary)
        rows.append(row)

    summary = pd.DataFrame(rows)
    out = (ARTIFACTS_DIR / "modeling" / "experiment_suite").resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "suite_summary.csv", index=False)

    ok = summary[summary["status"] == "ok"].copy()
    if not ok.empty:
        ok = ok.sort_values("mean_rmse")
        print("\n" + "=" * 100)
        print("CANONICAL EXPERIMENT SCOREBOARD — same data, same prefix grid, same LOBO CV")
        print("=" * 100)
        print(
            ok[
                [
                    "experiment_id",
                    "model",
                    "mean_rmse",
                    "std_rmse",
                    "median_rmse",
                    "worst_bearing_rmse",
                    "mean_mae",
                    "mean_r2",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )
    print(f"\nSuite summary: {out / 'suite_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
