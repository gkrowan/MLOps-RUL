"""Phase 18 drift simulation: POSTs engineered abnormal feature payloads
through the live /predict endpoint (real HTTP, not an in-process bypass),
then re-runs Phase 17's existing reporting pipeline to verify it flags
them. See docs/phase_18_drift_simulation.md for the full design.

Usage: python scripts/simulate_drift.py [--scenario amplitude|channel|quality|all]
                                         [--api-url http://localhost:8000]
                                         [--n-requests 15]
                                         [--out artifacts/monitoring/drift_simulation]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from femto_rul import config
from femto_rul.monitoring.current import load_current_features
from femto_rul.monitoring.drift_scenarios import (
    amplitude_drift,
    channel_drift,
    extreme_values,
    invalid_schema_requests,
    missing_value_sentinel,
)
from femto_rul.monitoring.reference import load_reference_features
from femto_rul.monitoring.report import build_report, drifted_column_share, save_report

WINDOW = "2 minutes"
# Conservative margin over WINDOW so back-to-back scenarios (--scenario all)
# don't pollute each other's evidence window.
SCENARIO_GAP_SECONDS = 130

# Fixed (not random) so evidence is reproducible run to run.
MISSING_VALUE_COLUMNS = [
    "rms_horiz_current_over_early",
    "kurtosis_vert_recent_mean_over_early",
    "crest_factor_horiz_recent_slope_per_hour",
]
EXTREME_VALUE_COLUMNS = ["observed_age_seconds", "rotation_speed_rpm", "radial_load_n"]


def _sample_base_rows(reference_df, n: int) -> list[dict]:
    sample = reference_df.sample(n=min(n, len(reference_df)))
    return sample.to_dict(orient="records")


def _post_payloads(api_url: str, payloads: list[dict]) -> None:
    for payload in payloads:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        if response.status_code != 200:
            print(f"  unexpected status {response.status_code}: {response.text[:200]}")


def _run_evidence_scenario(name: str, reference_df, out_dir: Path) -> float | None:
    print(f"  querying current window ({WINDOW}) for {name} evidence...")
    current_df = load_current_features(WINDOW)
    if current_df.empty:
        print(f"  no rows landed in the window for {name} — skipping report")
        return None

    snapshot = build_report(reference_df, current_df)
    share = drifted_column_share(snapshot)

    run_dir = out_dir / name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_report(snapshot, run_dir)
    print(f"  {name}: {len(current_df)} current rows, drifted_column_share={share}")
    print(f"  wrote report to {run_dir}")
    return share


def run_amplitude(api_url: str, n: int, reference_df, reference_ranges: dict, out_dir: Path):
    print("Scenario: amplitude drift")
    payloads = [
        amplitude_drift(row, reference_ranges) for row in _sample_base_rows(reference_df, n)
    ]
    _post_payloads(api_url, payloads)
    return _run_evidence_scenario("amplitude", reference_df, out_dir)


def run_channel(api_url: str, n: int, reference_df, out_dir: Path):
    print("Scenario: channel drift")
    payloads = [channel_drift(row) for row in _sample_base_rows(reference_df, n)]
    _post_payloads(api_url, payloads)
    return _run_evidence_scenario("channel", reference_df, out_dir)


def run_quality(api_url: str, n: int, reference_df, reference_ranges: dict, out_dir: Path):
    print("Scenario: data-quality anomaly")
    base_rows = _sample_base_rows(reference_df, n)
    payloads = [missing_value_sentinel(row, MISSING_VALUE_COLUMNS) for row in base_rows]
    payloads += [
        extreme_values(row, reference_ranges, EXTREME_VALUE_COLUMNS) for row in base_rows
    ]
    print("  posting missing-value-sentinel + extreme-value payloads...")
    _post_payloads(api_url, payloads)
    share = _run_evidence_scenario("quality", reference_df, out_dir)

    print("  posting invalid-schema payloads (expect 422, no predictions row)...")
    evidence = []
    for payload in invalid_schema_requests():
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        try:
            body = response.json()
        except ValueError:
            body = response.text
        evidence.append(
            {"payload": payload, "status_code": response.status_code, "response": body}
        )
        print(f"    status={response.status_code}")

    run_dir = out_dir / "schema_boundary" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(f"  wrote schema-boundary evidence to {run_dir}")

    return share


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=["amplitude", "channel", "quality", "all"], default="all"
    )
    parser.add_argument("--api-url", default=config.API_BASE_URL or "http://localhost:8000")
    parser.add_argument("--n-requests", type=int, default=15)
    parser.add_argument(
        "--out", default=str(config.MONITORING_ARTIFACTS_DIR / "drift_simulation")
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    api_url = args.api_url.rstrip("/")

    print("Loading reference distribution (prefix_train_v1.parquet)...")
    reference_df = load_reference_features()
    reference_ranges = json.loads(config.MONITORING_REFERENCE_RANGES_PATH.read_text())

    scenarios = (
        ["amplitude", "channel", "quality"] if args.scenario == "all" else [args.scenario]
    )
    shares: dict[str, float | None] = {}

    for i, scenario in enumerate(scenarios):
        if scenario == "amplitude":
            shares[scenario] = run_amplitude(
                api_url, args.n_requests, reference_df, reference_ranges, out_dir
            )
        elif scenario == "channel":
            shares[scenario] = run_channel(api_url, args.n_requests, reference_df, out_dir)
        elif scenario == "quality":
            shares[scenario] = run_quality(
                api_url, args.n_requests, reference_df, reference_ranges, out_dir
            )

        if i < len(scenarios) - 1:
            print(f"  waiting {SCENARIO_GAP_SECONDS}s so the next scenario's window is clean...")
            time.sleep(SCENARIO_GAP_SECONDS)

    print("\nSummary:")
    for scenario, share in shares.items():
        print(f"  {scenario}: drifted_column_share={share}")

    if not shares:
        sys.exit(1)


if __name__ == "__main__":
    main()
