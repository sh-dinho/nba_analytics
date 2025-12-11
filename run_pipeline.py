# ============================================================
# File: run_pipeline.py
# Purpose: Orchestrator for NBA prediction pipeline
# Version: 1.1 (Refactored)
# ============================================================

import argparse
from pathlib import Path
import yaml
import logging
from datetime import date

from prediction_engine.daily_runner_mlflow import daily_runner_mlflow
from src.model_training.train_combined import train_model

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------
# Load config safely
# -----------------------------
try:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    logging.error("Missing config.yaml file. Please create one before running.")
    exit(1)

MODEL_PATH = cfg.get("model_path")
MODEL_TYPE = cfg.get("model_type", "logreg")  # fallback to logistic regression
CACHE_PATH = Path("data/cache/features_full.parquet")

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Run NBA prediction pipeline")
parser.add_argument(
    "--train",
    action="store_true",
    help="Train model instead of running daily predictions",
)
parser.add_argument("--date", type=str, help="YYYY-MM-DD for daily run (default today)")
parser.add_argument(
    "--model-type", type=str, help="Override model type (default from config)"
)
parser.add_argument(
    "--save",
    action="store_true",
    help="Save predictions to CSV instead of just printing",
)
args = parser.parse_args()

# -----------------------------
# Main logic
# -----------------------------
if args.train:
    logging.info("Starting model training...")
    train_model(
        CACHE_PATH, out_dir=Path("models"), model_type=args.model_type or MODEL_TYPE
    )
    logging.info("Training complete. Model saved to models/")
else:
    # Resolve date
    date_str = args.date or date.today().strftime("%Y-%m-%d")

    # Validate model path
    if not MODEL_PATH or not Path(MODEL_PATH).exists():
        logging.error(f"Model path {MODEL_PATH} not found. Please check config.yaml.")
        exit(1)

    logging.info(f"Running daily NBA predictions for {date_str}...")
    df = daily_runner_mlflow(MODEL_PATH, game_date=date_str)

    if args.save:
        out_file = Path(f"predictions_{date_str}.csv")
        df.to_csv(out_file, index=False)
        logging.info(f"Predictions saved to {out_file}")
    else:
        logging.info("Predictions completed. Sample output:")
        print(df.head())
