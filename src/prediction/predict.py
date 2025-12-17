#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction and ranking for NBA games
Author: Mohamadou
"""

import logging
import pandas as pd

# Initialize logger
logger = logging.getLogger(__name__)


def generate_predictions(schedule_df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Generate win predictions based on a model or fallback logic using rolling features.

    Args:
        schedule_df (pd.DataFrame): DataFrame with schedule and features
        model (sklearn model, optional): Pre-trained model for predictions

    Returns:
        pd.DataFrame: DataFrame with predictions and rankings
    """
    df = schedule_df.copy()

    if model is None:  # Fallback logic if no model is provided
        if "HOME_AVG_LAST5PTS" not in df or "AWAY_AVG_LAST5PTS" not in df:
            logger.warning("Rolling features missing. Cannot generate predictions.")
            return df

        # If all rolling stats are NaN, skip prediction
        if df[["HOME_AVG_LAST5PTS", "AWAY_AVG_LAST5PTS"]].isna().all().all():
            logger.warning("No valid rolling stats for prediction. Skipping.")
            return df

        # Simple prediction based on recent performance (last 5 games)
        df["predicted_home_win"] = (
            df["HOME_AVG_LAST5PTS"] > df["AWAY_AVG_LAST5PTS"]
        ).astype(float)
        df["predicted_away_win"] = 1 - df["predicted_home_win"]

    else:  # Model-based prediction
        required_columns = [
            "HOME_AVG_LAST5PTS",
            "AWAY_AVG_LAST5PTS",
        ]  # You can expand this list with more features

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns for model: {missing_cols}")
            return df

        # Prepare features (fill missing values with 0)
        x = df[required_columns].fillna(0)

        try:
            # Make predictions
            predictions = model.predict(x)
            df["predicted_home_win"] = predictions
            df["predicted_away_win"] = 1 - predictions

            logger.info(
                f"Predictions made for {len(df)} games using the provided model."
            )
        except Exception as e:
            logger.error(
                f"Error during model prediction: {e}. Falling back to simple prediction logic."
            )
            # Recursively fallback to simple prediction logic
            return generate_predictions(df, model=None)

    # Rank games by predicted home win probability
    df["rank"] = df["predicted_home_win"].rank(ascending=False, method="dense")

    logger.info("Predictions and rankings added to schedule.")
    return df
