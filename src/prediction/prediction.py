#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction and ranking for NBA games
Author: Mohamadou
"""

import logging
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger(__name__)


def generate_predictions(schedule_df):
    """
    Generate simple win predictions based on rolling features.
    This can later be replaced by XGBoost/MLflow models.
    """
    df = schedule_df.copy()
    if "HOME_AVG_LAST5PTS" not in df or "AWAY_AVG_LAST5PTS" not in df:
        logger.warning("Rolling features missing. Cannot generate predictions.")
        return df

    # If all rolling stats are None, skip prediction
    if df[["HOME_AVG_LAST5PTS", "AWAY_AVG_LAST5PTS"]].isna().all().all():
        logger.warning("No valid rolling stats for prediction. Skipping.")
        return df

    df["predicted_home_win"] = (
        df["HOME_AVG_LAST5PTS"] > df["AWAY_AVG_LAST5PTS"]
    ).astype(float)
    df["predicted_away_win"] = 1 - df["predicted_home_win"]

    logger.info("Predictions added to schedule.")
    return df
