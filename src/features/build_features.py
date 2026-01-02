from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Script: Build Features
# File: src/scripts/build_features.py
# Author: Sadiq
#
# Description:
#     Loads the canonical long snapshot, builds the full
#     feature matrix using the feature pipeline, persists the
#     snapshot, and runs validation.
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import LONG_SNAPSHOT, FEATURES_SNAPSHOT
from src.features.feature_pipeline import build_features
from src.scripts.validate_features import validate_features


def main():
    logger.info("=== 🚀 Starting Feature Build ===")

    # --------------------------------------------------------
    # Load canonical long-format dataset
    # --------------------------------------------------------
    if not LONG_SNAPSHOT.exists():
        logger.error(f"❌ Missing canonical snapshot: {LONG_SNAPSHOT}")
        return

    logger.info(f"📥 Loading canonical long snapshot: {LONG_SNAPSHOT}")
    long_df = pd.read_parquet(LONG_SNAPSHOT)
    logger.info(f"Loaded {len(long_df)} team-game rows.")

    # --------------------------------------------------------
    # Build features + persist
    # --------------------------------------------------------
    logger.info("🔧 Running feature pipeline...")
    features = build_features(long_df, persist=True)

    logger.success(
        f"🎉 Feature build complete! Saved to {FEATURES_SNAPSHOT} "
        f"({features.shape[0]} rows, {features.shape[1]} columns)"
    )

    # --------------------------------------------------------
    # Validate the feature snapshot
    # --------------------------------------------------------
    logger.info("🛡️ Running post-build validation...")
    validate_features()

    logger.success("✨ Feature build + validation finished successfully.")


if __name__ == "__main__":
    main()
