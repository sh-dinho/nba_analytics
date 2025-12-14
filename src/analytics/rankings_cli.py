# ============================================================
# File: src/analytics/rankings_cli.py
# Purpose: CLI for generating team rankings and betting recommendations
# ============================================================

import logging
import click
from pathlib import Path
import pandas as pd
from src.analytics.rankings import RankingsManager

logger = logging.getLogger("analytics.rankings_cli")
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--predictions",
    required=True,
    help="Path to predictions CSV file (e.g., predictions_2025-12-12.csv)",
)
@click.option(
    "--out",
    default="data/results/rankings.csv",
    help="Path to save rankings CSV",
)
def cli(predictions, out):
    """Generate team rankings and betting recommendations from predictions."""
    logger.info("Loading predictions from %s", predictions)

    pred_path = Path(predictions)
    if not pred_path.exists():
        logger.error("Predictions file not found at %s", pred_path)
        return

    df_preds = pd.read_csv(pred_path)

    # Run rankings manager
    try:
        manager = RankingsManager(df_preds)
        df_rankings = manager.generate_rankings()
    except Exception as e:
        logger.error("Rankings generation failed: %s", e)
        return

    # Save rankings
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df_rankings.to_csv(out, index=False)
    logger.info("Rankings saved to %s", out)


if __name__ == "__main__":
    cli()
