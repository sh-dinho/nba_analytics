# ============================================================
# File: src/prediction/predict.py
# Purpose: NBA Win Predictions
# ============================================================

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def run_predictions(engineered_schedule: pd.DataFrame, config) -> pd.DataFrame:
    """
    Generate win predictions for NBA games.

    Args:
        engineered_schedule (pd.DataFrame): Master schedule with engineered features
        config: pipeline configuration

    Returns:
        pd.DataFrame: schedule with predicted win probabilities
    """
    if engineered_schedule.empty:
        logger.warning("Engineered schedule is empty. No predictions generated.")
        return pd.DataFrame()

    df = engineered_schedule.copy()

    # Example prediction: Use simple historical win percentage
    # In real pipeline, this could be a ML model
    df["predicted_win"] = df["home_team_win_pct"].fillna(0.5)  # fallback 50%
    df["predicted_loss"] = 1 - df["predicted_win"]

    logger.info(f"Predictions generated for {len(df)} games.")
    return df
