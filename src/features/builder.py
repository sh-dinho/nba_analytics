"""
Feature Builder
---------------
Transforms long-format game data into ML-ready feature matrices.

Responsibilities:
- Add engineered columns (season, game_number, points_for, points_against, won)
- Validate required schema
- Build rolling features, lag features, and model inputs
"""

# ============================================================
# File: src/features/builder.py
# Purpose: Transform long-format data into ML-ready features
# ============================================================

from __future__ import annotations
import pandas as pd
from loguru import logger


class FeatureBuilder:
    """
    Transforms team-level long data into features.
    Uses rolling windows to capture form without data leakage.
    """

    def __init__(self, window: int = 10):
        self.window = window

    def build(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point to build features.
        """
        if df_long.empty:
            logger.warning("FeatureBuilder: Input DataFrame is empty.")
            return df_long

        # 1. Prepare and Sort
        df = df_long.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["team", "date"])

        # 2. Define Rolling Calculations
        logger.info(f"FeatureBuilder: Building features with window={self.window}...")
        group = df.groupby("team")

        # Basic Rolling Stats (Shifted by 1 to prevent leakage)
        df["rolling_win_rate"] = group["won"].transform(
            lambda x: x.rolling(window=self.window, min_periods=1).mean().shift(1)
        )
        df["rolling_points_for"] = group["points_for"].transform(
            lambda x: x.rolling(window=self.window, min_periods=1).mean().shift(1)
        )
        df["rolling_points_against"] = group["points_against"].transform(
            lambda x: x.rolling(window=self.window, min_periods=1).mean().shift(1)
        )

        # 3. Strength of Schedule (Opponent's Rolling Win Rate)
        # This helps the model distinguish between a "fake" 8-2 record and a "real" one
        team_win_rates = df[["game_id", "team", "rolling_win_rate"]].rename(
            columns={"team": "opponent", "rolling_win_rate": "opp_rolling_win_rate"}
        )

        df = df.merge(team_win_rates, on=["game_id", "opponent"], how="left")

        # 4. Clean up
        before_count = len(df)
        df = df.dropna(subset=["rolling_win_rate", "opp_rolling_win_rate"])

        logger.info(
            f"FeatureBuilder: Processed {before_count} rows -> {len(df)} feature rows."
        )
        return df

    def get_feature_names(self) -> list[str]:
        return [
            "is_home",
            "rolling_win_rate",
            "rolling_points_for",
            "rolling_points_against",
            "opp_rolling_win_rate",
        ]
