# ============================================================
# File: src/prediction_engine/daily_runner_cli.py
# Purpose: CLI for running daily predictions
# ============================================================

import logging
import click
import pandas as pd
from src.prediction_engine.daily_runner import run_daily_predictions

logger = logging.getLogger("prediction_engine.daily_runner_cli")
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--model",
    required=True,
    help="Path to trained model file (e.g., models/nba_xgb.pkl)",
)
@click.option("--season", required=True, type=int, help="Season year (e.g., 2025)")
@click.option("--limit", default=10, help="Limit number of games to predict")
@click.option(
    "--out", default="data/results/daily_predictions.csv", help="Output CSV path"
)
@click.option(
    "--fmt", default="csv", type=click.Choice(["csv", "parquet"]), help="Output format"
)
def cli(model, season, limit, out, fmt):
    logger.info("Starting daily runner...")

    try:
        features_df, predictions_df, player_info_df = run_daily_predictions(
            model=model, season=season, limit=limit
        )

        # Save predictions
        if predictions_df.empty:
            logger.warning("No predictions generated (no games today or API timeout).")
        else:
            if fmt == "csv":
                predictions_df.to_csv(out, index=False)
            else:
                predictions_df.to_parquet(out, index=False)
            logger.info("Predictions saved to %s", out)

        # Save player info
        player_info_out = "data/results/player_info.csv"
        if not player_info_df.empty:
            player_info_df.to_csv(player_info_out, index=False)
            logger.info("Player info saved to %s", player_info_out)

    except Exception as e:
        logger.error("Daily runner failed: %s", e)


if __name__ == "__main__":
    cli()
