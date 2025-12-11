# ============================================================
# File: run_pipeline.py
# Purpose: Orchestrator for NBA prediction pipeline
# Version: 1.0
# ============================================================

import argparse
from pathlib import Path
import yaml
import logging

from prediction_engine.daily_runner_mlflow import daily_runner_mlflow
from src.model_training.train_combined import train_model

# -----------------------------
# Load config
# -----------------------------
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH = cfg["model_path"]
MODEL_TYPE = cfg.get("model_type", "logreg")  # fallback to logistic regression
CACHE_PATH = Path("data/cache/features_full.parquet")

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Run NBA prediction pipeline")
parser.add_argument("--train", action="store_true", help="Train model instead of running daily predictions")
parser.add_argument("--date", type=str, help="YYYY-MM-DD for daily run (default today)")
args = parser.parse_args()

# -----------------------------
# Main logic
# -----------------------------
if args.train:
    logging.info("Starting model training...")
    train_model(CACHE_PATH, out_dir=Path("models"), model_type=MODEL_TYPE)
    logging.info(f"Training complete. Model saved to models/")
else:
    date_str = args.date or "today"
    logging.info(f"Running daily NBA predictions for {date_str}...")
    df = daily_runner_mlflow(MODEL_PATH, game_date=date_str)
    logging.info(f"Predictions completed. Sample output:")
    print(df.head())
