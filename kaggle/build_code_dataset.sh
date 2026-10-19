#!/usr/bin/env bash
# Rebuild the `ganrajlongadge/cropguard-v7-code` staging tree from the repo.
#
# Kaggle mounts a dataset built from a folder at the mount ROOT, so the tree is
# FLATTENED to the names the kernel drivers expect:
#   architecture/  training/  data_pipeline/  __init__.py
# (not model/architecture/ ...). The drivers also accept model/<sub>, but the
# flat layout is what every past version of this dataset used.
#
# Usage:  kaggle/build_code_dataset.sh [STAGE_DIR]   (default /tmp/v7code)
#         kaggle/launch.sh push-code                 (rebuild + push a version)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-/tmp/v7code}"
SLUG="ganrajlongadge/cropguard-v7-code"

rm -rf "$STAGE"
mkdir -p "$STAGE/data_pipeline"

# --- python packages (flat) ---
cp -r "$REPO/model/architecture" "$STAGE/architecture"
cp -r "$REPO/model/training"     "$STAGE/training"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
cat > "$STAGE/__init__.py" <<'PY'
# CropGuard v7 code dataset (flat layout: architecture/, training/).
PY

# --- data pipeline (run by absolute path, not imported) ---
cp -r "$REPO/data_pipeline/scripts" "$STAGE/data_pipeline/scripts"
cp -r "$REPO/data_pipeline/configs" "$STAGE/data_pipeline/configs"

# --- plantseg driver kept alongside as a reference artefact ---
cp "$REPO/kaggle/plantseg/cropguard-v7-plantseg.py" "$STAGE/kernel_plantseg_driver.py"

# --- dataset metadata ---
cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "CropGuard v7 Code",
  "id": "$SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

echo "staged $STAGE:"
find "$STAGE" -type f | sed "s|$STAGE/||" | sort
