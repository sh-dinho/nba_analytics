# ============================================================
# File: features/feature_builder.py
# Purpose: Feature engineering for NBA Analytics
# ============================================================

import pandas as pd
from core.log_config import init_global_logger
from core.paths import (
    HISTORICAL_GAMES_FILE,
    NEW_GAMES_FILE,
    TRAINING_FEATURES_FILE,
    NEW_GAMES_FEATURES_FILE,
)

logger = init_global_logger()
logger.info("Feature builder module loaded.")

def build_features(training: bool, player: bool):
    """
    Builds features for training or prediction.

    Parameters
    ----------
    training : bool
        Flag to indicate if training features are being built.
    player : bool
        Flag to indicate if player features are being built.
    """
    logger.info(f"Building features: training={training}, player={player}")

    if training:
        logger.info("Building training features...")
        try:
            df = pd.read_csv(HISTORICAL_GAMES_FILE)

            # Example engineered feature
            if "PTS" in df.columns:
                df["point_diff"] = df["PTS"] - df["PTS"].shift(1, fill_value=0)

            # ✅ Add target column for training
            if "PTS_home" in df.columns and "PTS_away" in df.columns:
                df["home_win"] = (df["PTS_home"] > df["PTS_away"]).astype(int)
            elif "WL" in df.columns:  # NBA API often has Win/Loss flag
                df["home_win"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)
            else:
                logger.warning("⚠️ No explicit score columns found; target may be missing.")

            # Save training features
            df.to_csv(TRAINING_FEATURES_FILE, index=False)
            logger.info(f"Training features saved → {TRAINING_FEATURES_FILE}")
        except Exception as e:
            logger.error(f"❌ Failed to build training features: {e}")

    if not training:
        logger.info("Building upcoming game features...")
        try:
            df = pd.read_csv(NEW_GAMES_FILE)

            # Example placeholder engineered feature
            df["home_advantage"] = 1

            # Save new game features
            df.to_csv(NEW_GAMES_FEATURES_FILE, index=False)
            logger.info(f"Upcoming game features saved → {NEW_GAMES_FEATURES_FILE}")
        except Exception as e:
            logger.error(f"❌ Failed to build upcoming game features: {e}")

    if player:
        logger.info("Building player features...")
        # Add player-specific feature engineering here
        # Example placeholder
        # if "PTS" in df.columns and "MIN" in df.columns:
        #     df["player_efficiency"] = df["PTS"] / (df["MIN"] + 1)

    logger.info("Feature building completed.")