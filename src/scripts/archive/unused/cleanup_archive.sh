#!/bin/bash

echo "=============================================="
echo " NBA Analytics v3 — Full Repository Cleanup"
echo "=============================================="

ARCHIVE_DIR="archive/unused"
mkdir -p "$ARCHIVE_DIR"

echo "Archive directory created at $ARCHIVE_DIR"
echo

# ------------------------------------------------------------
# Define ALL files and folders that belong to v3 architecture
# ------------------------------------------------------------
KEEP_LIST=(
    # Root-level files
    "app.py"
    "config.py"
    "requirements.txt"
    "README.md"

    # Scripts
    "scripts/cleanup_archive.sh"

    # Ingestion
    "src/ingestion/collector.py"
    "src/ingestion/normalizer.py"
    "src/ingestion/pipeline.py"

    # Features
    "src/features/builder.py"
    "src/features/feature_store.py"

    # Model
    "src/model/train.py"
    "src/model/predict.py"

    # Monitoring
    "src/monitoring/drift.py"
    "src/monitoring/metrics_exporter.py"

    # Pipeline Orchestrator
    "src/pipeline/orchestrator.py"

    # Data directories (keep structure)
    "data"
    "data/ingestion"
    "data/features"
    "data/predictions"
    "data/models"
    "data/models/registry"
    "data/raw"
    "data/parquet"

    # Archive folder itself
    "archive"
)

# ------------------------------------------------------------
# Build a lookup file for quick matching
# ------------------------------------------------------------
KEEP_FILE=".keep_paths.tmp"
rm -f "$KEEP_FILE"
touch "$KEEP_FILE"

for item in "${KEEP_LIST[@]}"; do
    echo "$item" >> "$KEEP_FILE"
done

# ------------------------------------------------------------
# Helper: check if a path should be kept
# ------------------------------------------------------------
should_keep() {
    local path="$1"
    grep -Fxq "$path" "$KEEP_FILE"
}

# ------------------------------------------------------------
# Function: archive a file or folder
# ------------------------------------------------------------
archive_item() {
    local item="$1"
    local dest="$ARCHIVE_DIR/$item"

    mkdir -p "$(dirname "$dest")"
    mv "$item" "$dest" 2>/dev/null

    echo "ARCHIVED → $item"
}

# ------------------------------------------------------------
# Sweep through ALL files and folders in the repo
# ------------------------------------------------------------
echo "Scanning repository..."
echo

find . -mindepth 1 -maxdepth 5 | while read item; do
    # Normalize path (remove leading ./)
    CLEAN_PATH="${item#./}"

    # Skip archive folder itself
    if [[ "$CLEAN_PATH" == archive* ]]; then
        continue
    fi

    # Skip .git and hidden system files
    if [[ "$CLEAN_PATH" == .git* ]] || [[ "$CLEAN_PATH" == .* ]]; then
        continue
    fi

    # Skip directories that are part of KEEP_LIST
    if should_keep "$CLEAN_PATH"; then
        echo "KEEP     → $CLEAN_PATH"
        continue
    fi

    # Skip directories that contain kept files
    for keep in "${KEEP_LIST[@]}"; do
        if [[ "$keep" == "$CLEAN_PATH"* ]]; then
            echo "KEEP DIR → $CLEAN_PATH"
            continue 2
        fi
    done

    # Archive everything else
    archive_item "$CLEAN_PATH"
done

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
rm -f "$KEEP_FILE"

echo
echo "=============================================="
echo " Cleanup complete."
echo " All unused files moved to: $ARCHIVE_DIR"
echo "=============================================="
