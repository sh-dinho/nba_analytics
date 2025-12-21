"""
Feature Builder
---------------
Transforms long-format game data into ML-ready feature matrices.

Responsibilities:
- Add engineered columns (season, game_number, points_for, points_against, won)
- Validate required schema
- Build rolling features, lag features, and model inputs
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


class FeatureBuilder:
    """
    Converts long-format canonical data into ML-ready features.
    """

    REQUIRED_COLUMNS = {
        "game_id",
        "date",
        "team",
        "opponent",
        "is_home",
        "score",
        "opponent_score",
        "season",
        "game_number",
        "points_for",
        "points_against",
        "won",
    }

    def build(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point.
        """
        logger.info("FeatureBuilder: preparing long-format data...")
        df_long = self._prepare(df_long)

        logger.info("FeatureBuilder: validating schema...")
        self._validate_input(df_long)

        logger.info("FeatureBuilder: building features...")
        features_df = self._build_features(df_long)

        logger.info(f"FeatureBuilder: built {len(features_df)} feature rows.")
        return features_df

    # ----------------------------------------------------------------------
    # PREPARATION: Add missing engineered columns
    # ----------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add required engineered columns if missing."""
        df = df.copy()

        # Season string
        if "season" not in df.columns:
            df["season"] = df["date"].apply(
                lambda d: (
                    f"{d.year}-{str(d.year+1)[-2:]}"
                    if d.month >= 10
                    else f"{d.year-1}-{str(d.year)[-2:]}"
                )
            )

        # Game number per team per season
        if "game_number" not in df.columns:
            df["game_number"] = (
                df.sort_values("date").groupby(["team", "season"]).cumcount() + 1
            )

        # Points for / against
        if "points_for" not in df.columns:
            df["points_for"] = df["score"]

        if "points_against" not in df.columns:
            df["points_against"] = df["opponent_score"]

        # Win flag
        if "won" not in df.columns:
            df["won"] = (df["points_for"] > df["points_against"]).astype(int)

        return df

    # ----------------------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------------------

    def _validate_input(self, df: pd.DataFrame):
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"FeatureBuilder: missing required columns: {missing}")

    # ----------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ----------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build ML features:
        - Rolling averages
        - Lag features
        - Home/away indicator
        - Opponent strength features
        """
        df = df.copy()
        df = df.sort_values(["team", "date"])

        # Rolling stats (last 5 games for points, last 10 for win rate)
        df["rolling_points_for"] = (
            df.groupby("team")["points_for"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling_points_against"] = (
            df.groupby("team")["points_against"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling_win_rate"] = (
            df.groupby("team")["won"]
            .rolling(window=10, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Lag features (previous game)
        df["lag_points_for"] = df.groupby("team")["points_for"].shift(1)
        df["lag_points_against"] = df.groupby("team")["points_against"].shift(1)
        df["lag_won"] = df.groupby("team")["won"].shift(1)

        # Opponent rolling strength (simple season-aggregated stats)
        opp_stats = df.groupby("team").agg(
            opp_avg_points_for=("points_for", "mean"),
            opp_avg_points_against=("points_against", "mean"),
            opp_win_rate=("won", "mean"),
        )

        df = df.merge(
            opp_stats,
            left_on="opponent",
            right_index=True,
            how="left",
        )

        # Fill missing lag features with neutral values (early games)
        for col in ["lag_points_for", "lag_points_against", "lag_won"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # If opponent stats are missing (very early or odd data), fill with team-level averages
        for col in ["opp_avg_points_for", "opp_avg_points_against", "opp_win_rate"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        # Final feature set
        feature_cols = [
            "game_id",
            "date",
            "team",
            "opponent",
            "is_home",
            "season",
            "game_number",
            "points_for",
            "points_against",
            "won",
            "rolling_points_for",
            "rolling_points_against",
            "rolling_win_rate",
            "lag_points_for",
            "lag_points_against",
            "lag_won",
            "opp_avg_points_for",
            "opp_avg_points_against",
            "opp_win_rate",
        ]

        existing_cols = [c for c in feature_cols if c in df.columns]
        return df[existing_cols]
