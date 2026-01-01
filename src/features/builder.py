from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Feature Builder
# File: src/features/builder.py
# Author: Sadiq
#
# Description:
#     Build model-ready features from the canonical long
#     snapshot. This is the single entry point for feature
#     construction for all models.
# ============================================================

from dataclasses import dataclass
import pandas as pd
from loguru import logger


@dataclass
class FeatureBuilder:
    """
    Unified feature builder.

    The version field allows you to evolve feature sets over time
    without changing the public API or file names.
    """
    version: str = "default"

    def build(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point: build features from canonical long-format data.

        Required columns (minimum):
            - game_id
            - date
            - team
            - is_home
            - opponent
            - team_score
            - opp_score
            - season
        """
        logger.info(f"FeatureBuilder(version={self.version}): building features...")

        if self.version == "default":
            return self._build_default(long_df)

        raise ValueError(f"Unsupported feature version: {self.version}")

    # --------------------------------------------------------
    # Default feature set (formerly v5)
    # --------------------------------------------------------
    def _build_default(self, long_df: pd.DataFrame) -> pd.DataFrame:
        df = long_df.copy()

        # Basic sanity checks
        required = {"game_id", "team", "is_home", "opponent", "date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"FeatureBuilder: missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])

        # Sort for rolling features
        df = df.sort_values(["team", "date"])

        # Rolling win percentage
        if {"team_score", "opp_score"}.issubset(df.columns):
            df["win"] = (df["team_score"] > df["opp_score"]).astype(int)
            df["team_win_pct_last10"] = (
                df.groupby("team")["win"]
                .rolling(window=10, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        else:
            df["team_win_pct_last10"] = 0.5

        # Home indicator
        df["is_home_feature"] = df["is_home"].astype(int)

        # Opponent rolling win percentage
        if "team_win_pct_last10" in df.columns:
            opp_stats = (
                df[["date", "team", "team_win_pct_last10"]]
                .rename(columns={"team": "opp", "team_win_pct_last10": "opp_win_pct_last10"})
            )
            df = df.merge(
                opp_stats,
                left_on=["date", "opponent"],
                right_on=["date", "opp"],
                how="left",
            )
            df["opp_win_pct_last10"].fillna(0.5, inplace=True)
            df.drop(columns=["opp"], inplace=True)

        # Final feature set
        feature_cols = [
            "game_id",
            "team",
            "opponent",
            "is_home_feature",
            "team_win_pct_last10",
            "opp_win_pct_last10",
        ]

        features = (
            df[feature_cols]
            .drop_duplicates(subset=["game_id", "team"])
            .reset_index(drop=True)
        )

        logger.info(
            f"FeatureBuilder: built features shape={features.shape}, version={self.version}"
        )
        return features