"""Backward-compatible entry point for production feature extraction.

Historically this script created one combined labeled ``features.parquet`` that
mixed Training_set, Validation_Set, and labeled Test_set rows. That artifact is
unsafe for production training.

This entry point now delegates to ``build_datasets.py --mode all`` and creates:
- train_features.parquet
- test_features.parquet
- test_ground_truth.parquet
- feature_schema.json
"""

from build_datasets import main


if __name__ == "__main__":
    main()
