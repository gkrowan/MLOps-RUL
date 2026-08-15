#!/usr/bin/env python3
"""Run a controlled model comparison suite using the shared experiment runner."""

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
        help="Optional experiment IDs. Default: all configured experiments.",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config()
    ids = args.experiments or sorted(cfg.models)
    rows: list[dict[str, object]] = []

    for experiment_id in ids:
        spec = cfg.model(experiment_id)
        print(f"\n>>> {experiment_id}: {spec.model_name} [{spec.representation}]")
        try:
            result = run_experiment(experiment_id, track_mlflow=not args.no_mlflow, config=cfg)
        except MissingModelDependency as exc:
            print(f"SKIPPED: {exc}")
            rows.append({"experiment_id": experiment_id, "model": spec.model_name, "status": "skipped_missing_dependency"})
            continue
        row = {"experiment_id": experiment_id, "model": spec.model_name, "representation": spec.representation, "status": "ok"}
        row.update(result.summary)
        rows.append(row)

    summary = pd.DataFrame(rows)
    out = (ARTIFACTS_DIR / "modeling" / "experiment_suite").resolve()
    out.mkdir(parents=True, exist_ok=True)
    label = f"{ids[0]}-{ids[-1]}" if ids else "empty"
    summary_path = out / f"suite_summary_{label}.csv"
    summary.to_csv(summary_path, index=False)

    ok = summary[summary["status"] == "ok"].copy()
    if not ok.empty:
        ok = ok.sort_values("mean_rmse")
        print("\n" + "=" * 108)
        print("EXPERIMENT SCOREBOARD — same prefix grid, same direct-RUL target, same LOBO CV")
        print("=" * 108)
        print(ok[["experiment_id", "representation", "model", "mean_rmse", "std_rmse", "median_rmse", "worst_bearing_rmse", "mean_mae", "mean_r2"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSuite summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
