from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Feature Pipeline
# File: src/features/feature_pipeline.py
# Author: Sadiq
#
# Description:
#     Full end‑to‑end feature construction pipeline.
#     Produces model‑ready rows validated by FeatureRow.
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import FEATURES_SNAPSHOT
from src.features.feature_schema import FeatureRow

# Feature modules (Pipeline A)
from src.features.elo import add_elo_features
from src.features.elo_rolling import add_elo_rolling_features
from src.features.rolling import add_rolling_features
from src.features.form import add_form_features
from src.features.rest import add_rest_features
from src.features.sos import add_sos_features
from src.features.opponent_adjusted import add_opponent_adjusted_features
from src.features.margin_features import add_margin_features
from src.features.win_streak import add_win_streak


# ------------------------------------------------------------
# Row-level Pydantic validation
# ------------------------------------------------------------
def _validate_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    validated = []

    for idx, row in df.iterrows():
        try:
            model = FeatureRow(**row.to_dict())
            validated.append(model.model_dump())
        except Exception as e:
            logger.error(f"❌ Feature validation error at row {idx}: {e}")
            logger.error(f"Row contents: {row.to_dict()}")
            raise

    return pd.DataFrame(validated)


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def build_features(long_df: pd.DataFrame, persist: bool = False) -> pd.DataFrame:
    logger.info(f"🚀 Building features for {len(long_df)} team-game rows...")

    df = long_df.copy()

    # --------------------------------------------------------
    # Ensure datetime + season
    # --------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = (
        df["date"].dt.year.astype(str)
        + "-"
        + (df["date"].dt.year + 1).astype(str)
    )

    # --------------------------------------------------------
    # 1. Base Elo
    # --------------------------------------------------------
    logger.info("📌 Step 1: Base Elo features")
    df = add_elo_features(df)

    # --------------------------------------------------------
    # 2. Basic rolling stats
    # --------------------------------------------------------
    logger.info("📌 Step 2: Basic rolling features")
    df = add_rolling_features(df)

    # Rename win_rolling_10 → team_win_pct_last10
    df.rename(columns={"win_rolling_10": "team_win_pct_last10"}, inplace=True)

    # --------------------------------------------------------
    # 3. Win streak
    # --------------------------------------------------------
    df = add_win_streak(df)

    # --------------------------------------------------------
    # 4. Advanced rolling (Elo + margin)
    # --------------------------------------------------------
    logger.info("📌 Step 3: Advanced rolling features")
    df = add_elo_rolling_features(df)
    df = add_margin_features(df)

    # --------------------------------------------------------
    # 5. Contextual features
    # --------------------------------------------------------
    logger.info("📌 Step 4: Contextual features")
    df = add_rest_features(df)
    df = add_form_features(df)
    df = add_sos_features(df)

    # --------------------------------------------------------
    # 6. Opponent-adjusted features
    # --------------------------------------------------------
    logger.info("📌 Step 5: Opponent-adjusted features")
    df = add_opponent_adjusted_features(df)

    # --------------------------------------------------------
    # 7. Schema validation
    # --------------------------------------------------------
    logger.info("📌 Step 6: Validating feature schema...")
    df = _validate_feature_rows(df)

    # --------------------------------------------------------
    # 8. Persist snapshot
    # --------------------------------------------------------
    if persist:
        FEATURES_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FEATURES_SNAPSHOT, index=False)
        logger.success(f"💾 Features persisted → {FEATURES_SNAPSHOT}")

    logger.success(f"🎉 Feature pipeline complete! Final shape: {df.shape}")
    return df