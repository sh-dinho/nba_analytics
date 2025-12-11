#!/usr/bin/env python
# ============================================================
# File: src/prediction_engine/daily_runner_cli.py
# Purpose: Command-line interface for daily NBA predictions
# Project: nba_analysis
# Version: 1.0
# ============================================================

import click
import logging
from src.prediction_engine.daily_runner_mflow import daily_runner_mlflow

@click.command()
@click.option("--model", required=True, help="Path to trained model file (.pkl)")
@click.option("--date", required=True, help="Game date in YYYY-MM-DD format")
@click.option("--config", default="config.yaml", help="Path to config.yaml for MLflow settings")
def cli(model, date, config):
    """CLI for running daily NBA predictions and logging to MLflow."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        predictions = daily_runner_mlflow(model_path=model, game_date=date, config_file=config)
        if predictions.empty:
            click.echo(f"❌ No games found for {date}")
        else:
            click.echo(f"✅ Predictions complete for {date}. Logged {len(predictions)} games.")
            click.echo(predictions.head().to_string(index=False))
    except Exception as e:
        logging.error(f"Daily runner failed: {e}")
        click.echo("❌ Prediction run failed. See logs for details.")

if __name__ == "__main__":
    cli()
