"""Generate configs/monitoring_reference_ranges.json from
data/processed/train_features.parquet (Phase 5/6 output).

1st/99th percentile per Feature Set V1 column, from Training_set — not a
hand-picked guess. Phase 17's monitoring and Phase 18's drift simulation
both read this file, so they agree on what "in range" means for a feature.

Usage: python scripts/generate_reference_ranges.py [--out configs/monitoring_reference_ranges.json]
"""

import argparse
import json
from pathlib import Path

from femto_rul.monitoring.reference import load_reference_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="configs/monitoring_reference_ranges.json")
    args = parser.parse_args()

    df = load_reference_features()

    ranges = {
        column: {
            "p01": float(df[column].quantile(0.01)),
            "p99": float(df[column].quantile(0.99)),
        }
        for column in df.columns
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ranges, indent=2))
    print(f"Wrote {len(ranges)} feature ranges to {out_path}")


if __name__ == "__main__":
    main()
