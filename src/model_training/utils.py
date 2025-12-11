# ============================================================
# File: src/model_training/utils.py
# Purpose: Utility functions for model loading and feature building
# ============================================================

import logging
import joblib
import pandas as pd

logger = logging.getLogger("model_training.utils")


# --- Load model ---
def load_model(path: str):
    """Load a trained model from disk using joblib."""
    try:
        model = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return model
    except FileNotFoundError:
        logger.error("Model file not found: %s", path)
        raise
    except Exception as e:
        logger.error("Failed to load model from %s: %s", path, e)
        raise


# --- Build features ---
def build_features(games_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Build enriched features for prediction from games DataFrame.
    Includes rolling averages, home/away flags, win streaks.
    """
    try:
        df = games_df.copy()

        # --- Home/Away flag ---
        df["IS_HOME"] = df["HOME_TEAM_ABBREVIATION"].apply(
            lambda x: 1 if pd.notna(x) else 0
        )

        # --- Rolling averages (points scored/allowed) ---
        if "PTS" in df.columns and "PTS_OPP" in df.columns:
            df["AVG_PTS_LAST3"] = df.groupby("HOME_TEAM_ABBREVIATION")["PTS"].transform(
                lambda x: x.shift().rolling(3, min_periods=1).mean()
            )
            df["AVG_PTS_ALLOWED_LAST3"] = df.groupby("HOME_TEAM_ABBREVIATION")[
                "PTS_OPP"
            ].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())

        # --- Win streaks ---
        if "WL" in df.columns:
            df["WIN_STREAK"] = df.groupby("HOME_TEAM_ABBREVIATION")["WL"].transform(
                lambda x: x.eq("W").astype(int).groupby(x.ne("W").cumsum()).cumsum()
            )

        # --- Season flag ---
        df["SEASON"] = season

        # --- Select numeric features only ---
        features = df.select_dtypes(include=["number"]).fillna(0)

        logger.info("Features built (shape: %s)", features.shape)
        return features

    except Exception as e:
        logger.error("Failed to build features: %s", e)
        raise
