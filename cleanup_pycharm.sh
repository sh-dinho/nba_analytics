#!/bin/bash
# ============================================================
# Cleanup Script for NBA Analysis Project
# Purpose: Remove deprecated API, training, and feature modules after migration
# ============================================================

set -e

echo "[INFO] Starting cleanup..."

# Step 0: Ensure src directory exists
if [[ ! -d src ]]; then
  echo "[ERROR] src/ directory not found. Run this script from project root."
  exit 1
fi

# Step 0.5: Dry-run mode
DRY_RUN=${DRY_RUN:-false}
if [[ "$DRY_RUN" == "true" ]]; then
  echo "[INFO] Running in dry-run mode. No files will be deleted."
fi

# Step 1: Verify no lingering imports
echo "[INFO] Checking for references to deprecated modules..."
grep -R "nba_api" src/ || true
grep -R "nba_api_wrapper" src/ || true
grep -R "train_logreg" src/ || true
grep -R "train_xgb" src/ || true
grep -R "training" src/ || true
grep -R "daily_runner_mflow" src/ || true
grep -R "game_features" src/ || true

echo "[INFO] If you see results above, update those imports to use src.api.nba_api_client, src.model_training.train_combined, src.prediction_engine.daily_runner_cli, or src.features.feature_engineering"

# Step 2: Remove old files
remove_file() {
  local file=$1
  if [[ -f "$file" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
      echo "[DRY-RUN] Would remove $file"
    else
      rm -f "$file"
      echo "[INFO] Removed $file"
    fi
  fi
}

echo "[INFO] Removing deprecated API modules..."
remove_file src/api/nba_api.py
remove_file src/api/nba_api_wrapper.py

echo "[INFO] Removing deprecated training modules..."
remove_file src/model_training/train_logreg.py
remove_file src/model_training/train_xgb.py
remove_file src/model_training/training.py

echo "[INFO] Removing deprecated daily runner stub..."
remove_file src/prediction_engine/daily_runner_mflow.py

echo "[INFO] Removing deprecated feature generator..."
remove_file src/prediction_engine/game_features.py

# Step 3: Confirm removal
if [[ ! -f src/api/nba_api.py && ! -f src/api/nba_api_wrapper.py && \
      ! -f src/model_training/train_logreg.py && ! -f src/model_training/train_xgb.py && \
      ! -f src/model_training/training.py && \
      ! -f src/prediction_engine/daily_runner_mflow.py && \
      ! -f src/prediction_engine/game_features.py ]]; then
  echo "[SUCCESS] Deprecated modules removed. Use src/api/nba_api_client.py, src/model_training/train_combined.py, src/prediction_engine/daily_runner_cli.py, and src/features/feature_engineering.py going forward."
else
  echo "[ERROR] Cleanup failed — some files still exist."
fi
