#!/bin/bash
# ============================================================
# File: run_today.sh
# Purpose: Automate daily NBA picks pipeline, generate schedule, and copy outputs
# Author: nba_analysis
# ============================================================

set -euo pipefail

# --- Config ---
PYTHON_ENV="nba_env"                     # conda environment name
PROJECT_DIR="$HOME/nba_project"
OUTPUT_DIR="$PROJECT_DIR/results"
PBI_DIR="$HOME/pbi_folder"               # where Power BI reads CSVs

# --- Activate Python environment ---
echo "[INFO] Activating Python environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$PYTHON_ENV"

# --- Navigate to project ---
cd "$PROJECT_DIR" || { echo "[ERROR] Project dir not found"; exit 1; }

# --- Ensure cache/output dirs exist ---
mkdir -p "$PROJECT_DIR/data/cache" "$OUTPUT_DIR"

# --- Generate today's schedule/features ---
echo "[INFO] Generating today's schedule/features..."
python -m src.scripts.generate_today_schedule || { echo "[ERROR] Schedule generation failed"; exit 1; }

# --- Run today's NBA picks pipeline ---
echo "[INFO] Running today's pipeline..."
python -m src.main_today \
    --schedule_file "$PROJECT_DIR/data/cache/schedule.parquet" \
    --model_path "$PROJECT_DIR/models/logreg.pkl" \
    --out_dir "$OUTPUT_DIR"

# --- Copy CSVs for Power BI ---
echo "[INFO] Copying outputs to Power BI folder..."
for f in todays_picks.csv bet_on.csv avoid.csv; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        cp "$OUTPUT_DIR/$f" "$PBI_DIR/"
        echo "[INFO] Copied $f to Power BI folder"
    else
        echo "[WARN] $f not found in $OUTPUT_DIR"
    fi
done

echo "[INFO] Daily NBA pipeline completed. Outputs ready for Power BI."
