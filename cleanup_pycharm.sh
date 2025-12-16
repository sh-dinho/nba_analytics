# ============================================================
# File: scripts/cleanup_old.sh
# Purpose: Delete duplicate/obsolete files after restructure
# Project: nba_analysis
# Version: 1.0 (cleanup script)
# ============================================================

#!/bin/bash

FILES_TO_DELETE=(
  "src/utils/io.py"
  "src/utils/data_cleaning.py"
  "src/utils/add_unique_id.py"
  "src/utils/mapping.py"
  "src/scripts/download_baseline_schedule.py"
  "src/scripts/refresh_schedule.py"
  "src/scripts/refresh_schedule_incremental.py"
  "src/scripts/enrich_schedule.py"
  "src/scripts/generate_today_schedule.py"
  "src/scripts/generate_historical_schedule.py"
  "src/scripts/compare_schedule.py"
  "src/scripts/generate_features.py"
  "src/analytics/rankings_cli.py"
  "src/analytics/shap_analysis.py"
  "src/analytics/compare_algorithms.py"
)

echo "Starting cleanup of old files..."

for file in "${FILES_TO_DELETE[@]}"; do
  if [ -f "$file" ]; then
    echo "Removing $file"
    rm -f "$file"
  else
    echo "Skipping $file (not found)"
  fi
done

echo "Cleanup complete ✅"
