#!/bin/bash
# ============================================================
# File: run_today.sh
# Purpose: Automate daily NBA picks pipeline, generate schedule, and copy outputs
# Author: nba_analysis
# ============================================================

# --- Config ---
PYTHON_ENV="/path/to/your/python/env"   # e.g., conda activate nba_env
PROJECT_DIR="/path/to/nba_project"
OUTPUT_DIR="$PROJECT_DIR/results"
PBI_DIR="/path/to/pbi_folder"           # where Power BI reads CSVs

# --- Activate Python environment ---
echo "[INFO] Activating Python environment..."
source $PYTHON_ENV

# --- Navigate to project ---
cd $PROJECT_DIR || exit

# --- Generate today's schedule/features ---
echo "[INFO] Generating today's schedule/features..."
python -m src.scripts.generate_today_schedule

# --- Run today's NBA picks pipeline ---
echo "[INFO] Running today's pipeline..."
python -m src.main_today \
    --schedule_file "$PROJECT_DIR/data/cache/schedule.parquet" \
    --model_path "$PROJECT_DIR/models/logreg.pkl" \
    --out_dir "$OUTPUT_DIR"

# --- Copy CSVs for Power BI ---
echo "[INFO] Copying outputs to Power BI folder..."
cp $OUTPUT_DIR/todays_picks.csv $PBI_DIR/ 2>/dev/null || true
cp $OUTPUT_DIR/bet_on.csv $PBI_DIR/ 2>/dev/null || true
cp $OUTPUT_DIR/avoid.csv $PBI_DIR/ 2>/dev/null || true

echo "[INFO] Daily NBA pipeline completed. Outputs ready for Power BI."
