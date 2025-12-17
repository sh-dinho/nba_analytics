# ============================================================
# File: src/analytics/cli.py
# Purpose: CLI for generating team rankings and betting recommendations
# Project: nba_analysis
# Version: 1.0 (merged analytics CLI)
# ============================================================

import sys
import click
import pandas as pd
from pathlib import Path
from src.analytics.rankings import RankingsManager
from src.utils.common import configure_logging

# Set up logging for this module
logger = configure_logging(name="analytics.cli")


@click.command()
@click.option("--predictions", required=True, help="Path to predictions CSV file")
@click.option(
    "--out-rankings",
    default="data/results/rankings.csv",
    help="Path to save rankings CSV",
)
@click.option(
    "--out-betting",
    default="data/results/betting.csv",
    help="Path to save betting recommendations CSV",
)
def cli(predictions, out_rankings, out_betting):
    """Generate team rankings and betting recommendations from predictions."""

    # Read the predictions CSV file
    pred_path = Path(predictions)
    if not pred_path.exists():
        logger.error("Predictions file not found at %s", pred_path)
        sys.exit(1)

    df_preds = pd.read_csv(pred_path)

    # Initialize the RankingsManager to process the predictions
    manager = RankingsManager()

    # Generate team rankings based on predictions
    df_rankings = manager.generate_rankings(df_preds)

    # Save the rankings to CSV
    Path(out_rankings).parent.mkdir(parents=True, exist_ok=True)
    df_rankings.to_csv(out_rankings, index=False)
    logger.info("Rankings saved to %s", out_rankings)

    # Generate betting recommendations based on rankings
    recs = manager.betting_recommendations(df_rankings)

    # Save the betting recommendations to CSV
    Path(out_betting).parent.mkdir(parents=True, exist_ok=True)
    with open(out_betting, "w", encoding="utf-8") as f:
        f.write("# Bet On Teams\n")
        recs["bet_on"].to_csv(f, index=False)
        f.write("\n# Avoid Teams\n")
        recs["avoid"].to_csv(f, index=False)

    logger.info("Betting recommendations saved to %s", out_betting)


if __name__ == "__main__":
    cli()
