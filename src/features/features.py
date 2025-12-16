"""
File: features.py
Path: src/features
Purpose: Prepare features for NBA game outcome prediction.
         Includes feature engineering from historical and upcoming game data.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ----------------------
# Prepare Features
# ----------------------
def prepare_features(
    historical_schedule: pd.DataFrame, upcoming_games: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Prepare features for ML model.

    Args:
        historical_schedule (pd.DataFrame): Historical games with scores and stats
        upcoming_games (pd.DataFrame, optional): Games to predict. Default None.

    Returns:
        pd.DataFrame: Features ready for training or prediction
    """
    if historical_schedule.empty:
        logger.warning("Historical schedule is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    # Copy to avoid modifying original
    df = historical_schedule.copy()

    # Example features: team average points, opponent average points, home/away
    # Compute rolling stats per team
    df["home_team_points"] = df["home_score"]
    df["away_team_points"] = df["away_score"]

    # Target: 1 if home team wins, 0 if away team wins
    df["target"] = (df["home_score"] > df["away_score"]).astype(int)

    # Rolling averages for last 5 games
    df = df.sort_values(by="date")
    teams = pd.concat(
        [
            df[["date", "home_team", "home_score"]],
            df[["date", "away_team", "away_score"]].rename(
                columns={"away_team": "home_team", "away_score": "home_score"}
            ),
        ]
    )

    rolling_stats = (
        teams.groupby("home_team")["home_score"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index()
    )
    rolling_stats.rename(columns={"home_score": "team_avg_points"}, inplace=True)

    df = df.merge(
        rolling_stats[["level_1", "team_avg_points"]],
        left_index=True,
        right_on="level_1",
        how="left",
    )
    df.drop(columns=["level_1"], inplace=True)

    # Add home/away flag
    df["is_home"] = 1  # All games in historical are home perspective

    # Drop unnecessary columns
    cols_to_drop = ["home_score", "away_score", "date", "home_team", "away_team"]
    df_features = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # For prediction on upcoming games
    if upcoming_games is not None:
        upcoming = upcoming_games.copy()
        upcoming["is_home"] = (upcoming["home_team"].notna()).astype(int)
        # Add placeholder for rolling stats (could be updated with real averages)
        upcoming["team_avg_points"] = 100  # default placeholder
        df_features = upcoming

    logger.info(
        f"Prepared features: {df_features.shape[0]} rows, {df_features.shape[1]} columns."
    )

    return df_features
