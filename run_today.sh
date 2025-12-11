#!/bin/bash
set -euo pipefail

PYTHON_ENV="nba_env"
PROJECT_DIR="$HOME/nba_project"
TODAY=$(date +%Y-%m-%d)
OUTPUT_DIR="$PROJECT_DIR/results/$TODAY"
PBI_DIR="$HOME/pbi_folder"

mkdir -p "$PROJECT_DIR/data/cache" "$OUTPUT_DIR" "$PBI_DIR"

LOG_FILE="$OUTPUT_DIR/run_today.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] Activating Python environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$PYTHON_ENV"

cd "$PROJECT_DIR" || { echo "[ERROR] Project dir not found"; exit 1; }

echo "[INFO] Generating today's schedule/features..."
python -m src.scripts.generate_today_schedule || { echo "[ERROR] Schedule generation failed"; exit 1; }

MODEL_PATH="$PROJECT_DIR/models/logreg.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model file not found at $MODEL_PATH"
    exit 1
fi

echo "[INFO] Running today's pipeline..."
python -m src.main_today \
    --schedule_file "$PROJECT_DIR/data/cache/schedule.parquet" \
    --model_path "$MODEL_PATH" \
    --out_dir "$OUTPUT_DIR"

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
