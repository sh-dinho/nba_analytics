# ============================================================
# File: src/features/feature_engineering.py
# Purpose: Unified feature engineering for NBA game outcome prediction.
# Version: 2.1
# Author: Mohamadou (consolidated version)
# Date: 2025-12-17
# ============================================================

import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Elo Rating System
# ------------------------------------------------------------
def compute_elo(
    df: pd.DataFrame, base_elo: int = 1500, k: float = 20, home_adv: float = 80
) -> pd.DataFrame:
    df = df.sort_values("startDate").copy()
    elo = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        h, a = row["homeTeam"], row["awayTeam"]
        elo.setdefault(h, base_elo)
        elo.setdefault(a, base_elo)

        exp_home = 1 / (1 + 10 ** ((elo[a] - (elo[h] + home_adv)) / 400))
        exp_away = 1 - exp_home

        score_home = row["homeWin"]
        score_away = 1 - score_home

        elo[h] += k * (score_home - exp_home)
        elo[a] += k * (score_away - exp_away)

        home_elos.append(elo[h])
        away_elos.append(elo[a])

    df["elo_home"] = home_elos
    df["elo_away"] = away_elos
    return df


# ------------------------------------------------------------
# Opponent-Adjusted Metrics
# ------------------------------------------------------------
def add_opponent_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    opp_strength = (
        df.groupby("homeTeam")["point_diff"]
        .mean()
        .add(df.groupby("awayTeam")["point_diff"].mean(), fill_value=0)
        / 2
    )
    df["home_opp_strength"] = df["awayTeam"].map(opp_strength)
    df["away_opp_strength"] = df["homeTeam"].map(opp_strength)
    df["adj_pointdiff_home"] = df["point_diff"] - df["home_opp_strength"]
    df["adj_pointdiff_away"] = -df["point_diff"] - df["away_opp_strength"]
    return df


# ------------------------------------------------------------
# Rolling Window Features
# ------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.sort_values("startDate").copy()
    df["home_lastN_winpct"] = df.groupby("homeTeam")["homeWin"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    df["home_lastN_pointdiff"] = df.groupby("homeTeam")["point_diff"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    df["away_lastN_winpct"] = df.groupby("awayTeam")["homeWin"].transform(
        lambda s: (1 - s).rolling(window, min_periods=1).mean()
    )
    df["away_lastN_pointdiff"] = df.groupby("awayTeam")["point_diff"].transform(
        lambda s: (-s).rolling(window, min_periods=1).mean()
    )
    return df


# ------------------------------------------------------------
# Internal: Historical Feature Engineering
# ------------------------------------------------------------
def _prepare_historical_features(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        logger.warning("Schedule DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df = schedule_df.copy()
    required_columns = ["homeTeam", "awayTeam", "homeScore", "awayScore", "startDate"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["homeScore"] = pd.to_numeric(df["homeScore"], errors="coerce").fillna(0)
    df["awayScore"] = pd.to_numeric(df["awayScore"], errors="coerce").fillna(0)
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    df = df.sort_values("startDate")

    df["homeWin"] = (df["homeScore"] > df["awayScore"]).astype(int)
    df["point_diff"] = df["homeScore"] - df["awayScore"]
    df["rest_days_home"] = df.groupby("homeTeam")["startDate"].diff().dt.days.fillna(0)
    df["rest_days_away"] = df.groupby("awayTeam")["startDate"].diff().dt.days.fillna(0)

    df = add_rolling_features(df, window=10)
    df = compute_elo(df)
    df = add_opponent_adjusted(df)

    df["home_team_encoded"] = df["homeTeam"].astype("category").cat.codes
    df["away_team_encoded"] = df["awayTeam"].astype("category").cat.codes
    return df


# ------------------------------------------------------------
# Public API: prepare_features
# ------------------------------------------------------------
def prepare_features(
    historical_schedule: pd.DataFrame, upcoming_games: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    hist = _prepare_historical_features(historical_schedule)

    feature_cols = [
        "home_team_encoded",
        "away_team_encoded",
        "point_diff",
        "rest_days_home",
        "rest_days_away",
        "home_lastN_winpct",
        "away_lastN_winpct",
        "home_lastN_pointdiff",
        "away_lastN_pointdiff",
        "elo_home",
        "elo_away",
        "home_opp_strength",
        "away_opp_strength",
        "adj_pointdiff_home",
        "adj_pointdiff_away",
    ]

    # TRAINING MODE
    if upcoming_games is None:
        cols_to_keep = feature_cols + ["homeWin"]
        if "startDate" in hist.columns:
            cols_to_keep.append("startDate")  # preserve startDate for monitoring
        features = hist[cols_to_keep].copy()
        logger.info(
            f"Prepared TRAINING features: {features.shape[0]} rows, {features.shape[1]} columns."
        )
        return features

    # PREDICTION MODE (unchanged)
    upcoming = upcoming_games.copy()
    required_upcoming = ["homeTeam", "awayTeam", "startDate"]
    missing_upcoming = [c for c in required_upcoming if c not in upcoming.columns]
    if missing_upcoming:
        raise ValueError(
            f"Missing required columns in upcoming_games: {missing_upcoming}"
        )

    upcoming["startDate"] = pd.to_datetime(upcoming["startDate"], errors="coerce")
    team_cat = pd.Categorical(pd.concat([hist["homeTeam"], hist["awayTeam"]]).unique())
    team_to_code = {team: code for code, team in enumerate(team_cat)}

    upcoming["home_team_encoded"] = (
        upcoming["homeTeam"].map(team_to_code).fillna(-1).astype(int)
    )
    upcoming["away_team_encoded"] = (
        upcoming["awayTeam"].map(team_to_code).fillna(-1).astype(int)
    )

    last_hist = (
        hist.sort_values("startDate").groupby("homeTeam").tail(1).set_index("homeTeam")
    )

    def map_or_default(team_col: pd.Series, col_name: str, default=0.0):
        return team_col.map(last_hist[col_name]).fillna(default)

    upcoming["home_lastN_winpct"] = map_or_default(
        upcoming["homeTeam"], "home_lastN_winpct"
    )
    upcoming["away_lastN_winpct"] = map_or_default(
        upcoming["awayTeam"], "away_lastN_winpct"
    )
    upcoming["home_lastN_pointdiff"] = map_or_default(
        upcoming["homeTeam"], "home_lastN_pointdiff"
    )
    upcoming["away_lastN_pointdiff"] = map_or_default(
        upcoming["awayTeam"], "away_lastN_pointdiff"
    )
    upcoming["elo_home"] = map_or_default(
        upcoming["homeTeam"], "elo_home", default=1500
    )
    upcoming["elo_away"] = map_or_default(
        upcoming["awayTeam"], "elo_away", default=1500
    )
    upcoming["home_opp_strength"] = map_or_default(
        upcoming["homeTeam"], "home_opp_strength"
    )
    upcoming["away_opp_strength"] = map_or_default(
        upcoming["awayTeam"], "away_opp_strength"
    )
    upcoming["adj_pointdiff_home"] = map_or_default(
        upcoming["homeTeam"], "adj_pointdiff_home"
    )
    upcoming["adj_pointdiff_away"] = map_or_default(
        upcoming["homeTeam"], "adj_pointdiff_away"
    )

    upcoming["point_diff"] = 0.0
    upcoming["rest_days_home"] = 0.0
    upcoming["rest_days_away"] = 0.0

    pred_features = upcoming[feature_cols + ["startDate"]].copy()
    logger.info(
        f"Prepared PREDICTION features: {pred_features.shape[0]} rows, {pred_features.shape[1]} columns."
    )
    return pred_features
