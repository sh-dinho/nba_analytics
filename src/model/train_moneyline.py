from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Moneyline Model Training Entrypoint
# File: src/model/train_moneyline.py
# Author: Sadiq
#
# Description:
#     Entrypoint script for training and registering a
#     moneyline (win probability) model using:
#       - canonical long snapshot
#       - FeatureBuilder v4
#       - training_core v4
#       - model registry v4
#
#     IMPORTANT:
#       training_core requires BOTH:
#         - raw long snapshot columns (score, opponent_score, etc.)
#         - feature columns from FeatureBuilder
#
#       So we merge features back onto the long snapshot.
# ============================================================

from loguru import logger
import pandas as pd

from src.config.paths import LONG_SNAPSHOT
from src.features.builder import FeatureBuilder
from src.model.training_core import train_and_register_model


def main():
    logger.info("🏀 Starting v4 moneyline model training")

    # --------------------------------------------------------
    # Load canonical long snapshot
    # --------------------------------------------------------
    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info(f"Loaded canonical long snapshot: rows={len(df)}")

    # --------------------------------------------------------
    # Build v4 features
    # --------------------------------------------------------
    fb = FeatureBuilder(version="v4")
    features_df = fb.build_from_long(df)

    logger.success(
        f"Features built: rows={len(features_df)}, cols={features_df.shape[1]}"
    )

    # --------------------------------------------------------
    # Merge features back onto long snapshot
    # training_core needs:
    #   - score, opponent_score
    #   - game_id, team, opponent, date
    #   - numeric feature columns
    # --------------------------------------------------------
    merged = df.merge(
        features_df,
        on=["game_id", "team", "date"],
        how="left",
        validate="many_to-one",
    )

    logger.info(f"Merged training frame: rows={len(merged)}, cols={merged.shape[1]}")

    # --------------------------------------------------------
    # Train + register model
    # --------------------------------------------------------
    train_and_register_model(
        model_type="moneyline",
        df=merged,
        feature_version="v4",
    )

    logger.success("🏀 Moneyline model trained and registered successfully")


if __name__ == "__main__":
    main()
