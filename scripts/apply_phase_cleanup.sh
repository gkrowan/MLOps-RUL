#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src/femto_rul" || ! -f "pyproject.toml" ]]; then
  echo "Run this script from the MLOps-RUL repository root." >&2
  exit 1
fi

# These placeholders are obsolete now that raw contains Git-tracked DVC metadata
# and the architecture no longer uses an interim extraction layer.
rm -f data/raw/.gitkeep
rm -f data/interim/.gitkeep
rmdir data/interim 2>/dev/null || true

echo "Cleanup complete. Review with: git status --short"
