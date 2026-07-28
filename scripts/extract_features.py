"""Run feature extraction across the whole FEMTO dataset and write the
resulting per-snapshot table to data/processed/features.parquet.

This is a batch job over ~40k raw files and takes on the order of 20
minutes single-threaded. data/processed/ is inside the gitignored data/
directory — this output stays local until pushed to the team's shared
drive, same as the raw dataset.

Usage: python scripts/extract_features.py [--data-dir data] [--out data/processed/features.parquet]
"""

import argparse
from pathlib import Path

from femto_rul.pipeline import build_full_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/processed/features.parquet")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Extracting features across Training_set, Validation_Set, Test_set...")
    df = build_full_dataset(Path(args.data_dir))
    print(f"Built {len(df)} rows, {len(df.columns)} columns")

    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
