#!/usr/bin/env bash
# ============================================================
# Script: cleanup_repo.sh
# Purpose: Remove redundant files from nba_analysis project
# ============================================================

set -e

echo "Cleaning up redundant files..."

# Prediction engine
rm -f src/prediction_engine/daily_runner.py

# Model training
rm -f src/model_training/train_logreg.py
rm -f src/model_training/train_xgb.py
rm -f src/model_training/train_combined.py

# Old CLIs (if present)
rm -f src/model_training/train_cli.py
rm -f src/prediction_engine/predictor_cli_old.py

# Pipeline skeleton
rm -f src/pipeline/run_pipeline.py

echo "Cleanup complete. Canonical files remain intact."
