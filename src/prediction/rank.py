# ============================================================
# File: src/prediction/rank.py
# Purpose: Rank teams based on predictions and generate betting decisions
# ============================================================

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def generate_rankings(predictions: pd.DataFrame):
    """
    Generate rankings and betting signals based on predictions.

    Args:
        predictions: DataFrame containing predicted_win column

    Returns:
        DataFrame with ranking and betting signals
    """
    if predictions.empty:
        logger.warning("No predictions available for ranking.")
        return pd.DataFrame()

    # Rank games by predicted win probability
    predictions["rank"] = predictions["predicted_win"].rank(
        method="min", ascending=False
    )

    # Generate simple betting signal (1 = bet for home team, 0 = no bet)
    predictions["bet_signal"] = (predictions["predicted_win"] > 0.6).astype(int)

    # Count summary stats
    num_games = len(predictions)
    num_bets = predictions["bet_signal"].sum()
    logger.info(
        f"Rankings generated for {num_games} games, {num_bets} betting signals created."
    )

    # Sort by rank
    rankings = predictions.sort_values("rank", ascending=True).reset_index(drop=True)

    return rankings
