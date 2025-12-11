# ============================================================
# File: src/prediction_engine/daily_runner_cli.py
# Purpose: CLI to run daily predictions and save results
# Project: nba_analysis
# Version: 2.0 (no tracker dependency)
# ============================================================

import click
import os
import pandas as pd
import logging

from src.prediction_engine.daily_runner import run_daily_predictions
from src.utils.logging_config import configure_logging

RESULTS_DIR = "data/results"
DEFAULT_FILE = os.path.join(RESULTS_DIR, "daily_predictions.parquet")


@click.command()
@click.option(
    "--model",
    required=True,
    help="Path to trained model file (e.g. models/nba_xgb.pkl)",
)
@click.option("--season", default=2025, help="NBA season year")
@click.option("--limit", default=10, help="Number of games to fetch")
@click.option("--out", default=DEFAULT_FILE, help="Output file path")
@click.option(
    "--fmt",
    default="parquet",
    type=click.Choice(["parquet", "csv"]),
    help="Output format",
)
def cli(model, season, limit, out, fmt):
    """Run daily predictions and save results to disk."""
    logger = configure_logging(name="prediction_engine.daily_runner_cli")
    logger.info("Starting daily runner...")

    features_df, predictions_df, player_info_df = run_daily_predictions(
        model_path=model, season=season, limit=limit
    )

    if features_df is None or predictions_df is None:
        logger.warning("No predictions generated today.")
        return

    os.makedirs(os.path.dirname(out), exist_ok=True)

    try:
        if fmt == "parquet":
            predictions_df.to_parquet(out, index=False)
        else:
            predictions_df.to_csv(out, index=False)
        logger.info("Predictions saved to %s (%s)", out, fmt.upper())
    except Exception as e:
        logger.error("Error saving predictions: %s", e)


if __name__ == "__main__":
    cli()
