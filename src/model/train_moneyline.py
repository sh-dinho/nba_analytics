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
    logger.info(f"Loaded canonical long snapshot: rows={len(df)}")

    # --------------------------------------------------------
    # Build v4 features
    # --------------------------------------------------------
    fb = FeatureBuilder(version="v4")
    features_df = fb.build_from_long(df)

    logger.success(
        f"Features built for training: rows={len(features_df)}, cols={features_df.shape[1]}"
    )

    # --------------------------------------------------------
    # Train + register model
    # --------------------------------------------------------
    train_and_register_model(
        model_type="moneyline",
        df=features_df,
        feature_version="v4",
    )

    logger.success("🏀 Moneyline model trained and registered successfully")


if __name__ == "__main__":
    main()
