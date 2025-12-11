# ============================================================
# File: src/run_pipeline.py
# Purpose: Command-line entry point to train models or run daily predictions
# Project: nba_analysis
# Version: 1.1 (adds dependencies section + clearer CLI handling)
#
# Dependencies:
# - argparse (standard library)
# - pathlib (standard library)
# - yaml
# - prediction_engine.daily_runner_mlflow
# - src.model_training.train_combined
# ============================================================

import argparse
from pathlib import Path

import yaml

from prediction_engine.daily_runner_mlflow import daily_runner_mlflow
from src.model_training.train_combined import train_model


def main():
    # -----------------------------
    # Load configuration
    # -----------------------------
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    MODEL_PATH = cfg.get("model_path")
    MODEL_TYPE = cfg.get("model_type")
    CACHE_PATH = Path("data/cache/features_full.parquet")

    # -----------------------------
    # CLI Arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Train model or run daily NBA predictions.")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--date", help="YYYY-MM-DD for daily run (default: today)")
    args = parser.parse_args()

    # -----------------------------
    # Execution
    # -----------------------------
    if args.train:
        train_model(CACHE_PATH, out_dir=Path("models"), model_type=MODEL_TYPE)
    else:
        date = args.date or "today"
        df = daily_runner_mlflow(MODEL_PATH, date)
        print(df.head())


if __name__ == "__main__":
    main()
