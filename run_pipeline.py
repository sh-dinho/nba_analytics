# ============================================================
# File: run_pipeline.py
# Purpose: Orchestrator for NBA prediction pipeline
# Version: 1.3 (finalized for end-to-end run)
# ============================================================

import argparse
from pathlib import Path
import logging
from datetime import date

from src.daily_runner.daily_runner_mlflow import daily_runner_mlflow
from src.model_training.train_combined import train_model
from src.config import load_config  # unified config loader

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
    cfg = load_config("config.yaml")
except Exception as e:
    logging.error(f"Failed to load config.yaml: {e}")
    exit(1)

MODEL_PATH = cfg.model.path
MODEL_TYPE = cfg.model.type or "logreg"  # fallback to logistic regression
CACHE_PATH = Path(cfg.paths.cache) / "features_full.parquet"

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
        CACHE_PATH,
        out_dir=Path(cfg.paths.models),
        model_type=args.model_type or MODEL_TYPE,
    )
    logging.info("Training complete. Model saved to %s", cfg.paths.models)
else:
    # Resolve date
    date_str = args.date or date.today().strftime("%Y-%m-%d")

    # Validate model path
    if not MODEL_PATH or not Path(MODEL_PATH).exists():
        logging.error(f"Model path {MODEL_PATH} not found. Please check config.yaml.")
        exit(1)

    logging.info(f"Running daily NBA predictions for {date_str}...")
    df = daily_runner_mlflow(MODEL_PATH, game_date=date_str)

    if df is None or df.empty:
        logging.warning("No predictions generated for %s", date_str)
        exit(0)

    if args.save:
        out_file = Path(f"predictions_{date_str}.csv")
        df.to_csv(out_file, index=False)
        logging.info(f"Predictions saved to {out_file}")
    else:
        logging.info("Predictions completed. Sample output:")
        print(df.head())
