from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Unified Model Training Entrypoint
# File: src/model/train_all_models.py
# Author: Sadiq
#
# Description:
#     Trains ALL v4 models in a single orchestrated run:
#
#       1. Moneyline (win probability)
#       2. Totals (over/under total points)
#       3. Spread regression (margin)
#       4. Spread classification (ATS cover) — optional
#
#     Each model:
#       - Uses FeatureBuilder v4
#       - Uses canonical long snapshot
#       - Saves model artifact
#       - Saves training-time schema
#       - Registers metadata in the v4 registry
#
#     Run:
#         python -m src.model.train_all_models
# ============================================================

from loguru import logger
import pandas as pd

from src.config.paths import LONG_SNAPSHOT
from src.features.builder import FeatureBuilder

# Individual trainers
from src.model.train_moneyline import main as train_moneyline
from src.model.train_totals import train_totals_model
from src.model.train_spread import train_spread_models


# ------------------------------------------------------------
# Load canonical long snapshot once
# ------------------------------------------------------------


def _load_long_snapshot() -> pd.DataFrame:
    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info(f"Loaded canonical long snapshot: rows={len(df)}")
    return df


# ------------------------------------------------------------
# Unified training orchestrator
# ------------------------------------------------------------


def train_all_models():
    logger.info("🏀 Starting unified v4 model training pipeline")

    # Load once
    df_long = _load_long_snapshot()

    # Precompute v4 features once
    logger.info("Building v4 features (shared across all models)")
    fb = FeatureBuilder(version="v4")
    features_df = fb.build_from_long(df_long)
    logger.success(
        f"v4 features built: rows={len(features_df)}, cols={features_df.shape[1]}"
    )

    # --------------------------------------------------------
    # MONEYLINE
    # --------------------------------------------------------
    try:
        logger.info("🏀 Training moneyline model")
        train_moneyline()  # uses its own merge logic
        logger.success("Moneyline model training complete")
    except Exception as e:
        logger.error(f"[Moneyline] Training failed: {e}")

    # --------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------
    try:
        logger.info("🏀 Training totals model")
        train_totals_model()
        logger.success("Totals model training complete")
    except Exception as e:
        logger.error(f"[Totals] Training failed: {e}")

    # --------------------------------------------------------
    # SPREAD (regression + optional ATS classification)
    # --------------------------------------------------------
    try:
        logger.info("🏀 Training spread models")
        train_spread_models()
        logger.success("Spread model training complete")
    except Exception as e:
        logger.error(f"[Spread] Training failed: {e}")

    logger.success("🏁 Unified v4 model training pipeline complete")


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------

if __name__ == "__main__":
    train_all_models()
