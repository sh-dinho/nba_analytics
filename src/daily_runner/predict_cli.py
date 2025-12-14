# ============================================================
# File: src/daily_runner/predict_cli.py
# Purpose: CLI for running daily predictions with a trained model
# ============================================================

import logging
import click
import os
from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime
from src.daily_runner.daily_runner_mlflow import daily_runner_mlflow

logger = logging.getLogger("daily_runner.predict_cli")
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--date",
    required=False,
    default=datetime.today().strftime("%Y-%m-%d"),
    help="Game date (YYYY-MM-DD). Defaults to today.",
)
@click.option(
    "--model",
    default="models/nba_model.pkl",
    help="Path to trained model file",
)
@click.option(
    "--out",
    default=None,
    help="Path to save predictions CSV (default: predictions_<date>.csv)",
)
def cli(date, model, out):
    """Run daily predictions for a given date using a trained model."""
    logger.info("Running predictions for date: %s", date)

    model_path = Path(model)
    if not model_path.exists():
        logger.error("Model file not found at %s. Train a model first.", model_path)
        return

    # Run prediction pipeline
    try:
        df_preds = daily_runner_mlflow(str(model_path), game_date=date)
    except Exception as e:
        logger.error("Prediction pipeline failed: %s", e)
        return

    if df_preds is None or df_preds.empty:
        logger.warning("No predictions generated for %s", date)
        return

    # Save predictions
    if out is None:
        out = f"predictions_{date}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_preds.to_csv(out, index=False)
    logger.info("Predictions saved to %s", out)


if __name__ == "__main__":
    cli()
