# ============================================================
# File: scripts/build_features.py
# Description:
#   NBA feature builder with enhancements:
#     - Faster rolling calculations
#     - Expanded features (streaks, home/away splits, player usage)
#     - Better error handling and logging
#     - Human-readable metadata
# ============================================================

import os
import sys
import logging
from datetime import datetime
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import json

# ----------------------------
# Configuration & schema
# ----------------------------
REQUIRED_GAMES_COLUMNS = {"game_id", "date", "home_team", "away_team", "home_points", "away_points"}
REQUIRED_PLAYER_COLUMNS = {"date", "player", "team", "points", "rebounds", "assists", "minutes"}
OPTIONAL_INJURY_COLUMNS = {"date", "team", "impact_minutes"}
OPTIONAL_ODDS_COLUMNS = {"game_id", "home_moneyline", "away_moneyline", "spread", "total"}

DEFAULT_DATA_DIR = "data"
DEFAULT_FEATURES_PATH = os.path.join(DEFAULT_DATA_DIR, "training_features.csv")

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("build_features")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

# ----------------------------
# IO helpers
# ----------------------------
def _safe_read(path: str, required: bool = True) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None
    df = pd.read_csv(path)
    logger.info(f"Loaded {path} with shape={df.shape}")
    return df

def _ensure_columns(df: pd.DataFrame, required_cols: set, name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def _parse_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    if df[cols[0]].isna().any():
        logger.warning(f"Found NaT in date column {cols[0]} after parsing")
    return df

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ----------------------------
# Feature helpers
# ----------------------------
def _rolling_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    cols: List[str],
    windows: Tuple[int, ...] = (3, 5, 10),
    shift_by_one: bool = True
) -> pd.DataFrame:
    df = df.sort_values([group_col, date_col]).copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    shifted = df.groupby(group_col)[cols].shift(1) if shift_by_one else df[cols]

    for w in windows:
        for c in cols:
            df[f"{c}_roll{w}"] = shifted.groupby(df[group_col])[c] \
                                        .transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"{c}_wroll{w}"] = shifted.groupby(df[group_col])[c] \
                                         .transform(lambda x: x.rolling(w, min_periods=1)
                                                    .apply(lambda arr: np.average(arr, weights=np.linspace(1, 2, len(arr))), raw=True))
    return df

def _team_opponent_strength(games: pd.DataFrame) -> pd.DataFrame:
    home = games.rename(columns={"home_team": "team", "away_team": "opponent",
                                 "away_points": "opp_points", "home_points": "points"})[['date', 'team', 'opponent', 'points', 'opp_points']]
    away = games.rename(columns={"away_team": "team", "home_team": "opponent",
                                 "home_points": "opp_points", "away_points": "points"})[['date', 'team', 'opponent', 'points', 'opp_points']]
    long = pd.concat([home, away], ignore_index=True)
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.sort_values(["team", "date"])
    long["points_against_opponent_shift"] = long.groupby("opponent")["points"].shift(1)
    long["opp_def_strength"] = long.groupby("opponent")["points_against_opponent_shift"] \
                                   .transform(lambda s: s.rolling(10, min_periods=3).mean())
    return long[["date", "team", "opponent", "opp_def_strength"]].copy()

def _injury_impact(inj: pd.DataFrame) -> pd.DataFrame:
    if "impact_minutes" not in inj.columns:
        inj["impact_minutes"] = 0
    inj["impact_minutes"] = pd.to_numeric(inj["impact_minutes"], errors="coerce").fillna(0)
    impact = inj.groupby(["date", "team"])["impact_minutes"].sum().reset_index()
    impact.rename(columns={"impact_minutes": "team_impact_minutes"}, inplace=True)
    return impact

def _rest_features(games: pd.DataFrame) -> pd.DataFrame:
    def team_schedule(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["date"])
        df = df.groupby("team")["date"].apply(lambda d: d.sort_values()).reset_index()
        df["prev_date"] = df.groupby("team")["date"].shift(1)
        df["rest_days"] = (df["date"] - df["prev_date"]).dt.days.fillna(5)
        df["b2b"] = (df["rest_days"] <= 1).astype(int)
        return df[["team", "date", "rest_days", "b2b"]]

    home = team_schedule(games[["date", "home_team"]].rename(columns={"home_team": "team"}))
    away = team_schedule(games[["date", "away_team"]].rename(columns={"away_team": "team"}))
    return pd.concat([home, away], ignore_index=True)

def _moneyline_to_decimal(ml: float) -> Optional[float]:
    if pd.isna(ml):
        return None
    try:
        ml = float(ml)
        if ml > 0:
            return 1.0 + (ml / 100.0)
        elif ml < 0:
            return 1.0 + (100.0 / abs(ml))
        else:
            return None
    except Exception as e:
        logger.warning(f"Invalid moneyline value {ml}: {e}")
        return None

# ----------------------------
# Expanded features
# ----------------------------
def _add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add win/loss streak features for each team.
    Expects columns: team, date, points, opp_points.
    """
    df = df.sort_values(["team", "date"]).copy()

    # Win indicator
    df["win"] = (df["points"] > df["opp_points"]).astype(int)

    # Loss indicator
    df["loss"] = 1 - df["win"]

    # Rolling win streak: consecutive wins until a loss
    df["win_streak"] = df.groupby("team")["win"].transform(
        lambda s: s * (s.groupby((s != s.shift()).cumsum()).cumsum())
    )

    # Rolling loss streak: consecutive losses until a win
    df["loss_streak"] = df.groupby("team")["loss"].transform(
        lambda s: s * (s.groupby((s != s.shift()).cumsum()).cumsum())
    )

    return df

def _home_away_splits(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team-level home/away splits:
    - avg points, rebounds, assists when playing home/away
    - rolling win percentages
    Leakage-safe by shifting.
    """
    # Build long-format games with home/away flag
    games_long = pd.concat([
        games[["date","home_team"]].rename(columns={"home_team":"team"}).assign(is_home=1),
        games[["date","away_team"]].rename(columns={"away_team":"team"}).assign(is_home=0)
    ])
    players = players.merge(games_long, on=["team","date"], how="left")

    # Win indicator (requires opp_points in players)
    if "opp_points" in players.columns:
        players["win"] = (players["points"] > players["opp_points"]).astype(int)

    agg_cols = ["points","rebounds","assists"]
    windows = (3,5,10)
    for w in windows:
        for col in agg_cols:
            players[f"{col}_home_roll{w}"] = players.loc[players["is_home"]==1].groupby("team")[col].transform(
                lambda x: x.shift(1).rolling(w,min_periods=1).mean()
            )
            players[f"{col}_away_roll{w}"] = players.loc[players["is_home"]==0].groupby("team")[col].transform(
                lambda x: x.shift(1).rolling(w,min_periods=1).mean()
            )
        if "win" in players.columns:
            players[f"win_pct_home_roll{w}"] = players.loc[players["is_home"]==1].groupby("team")["win"].transform(
                lambda x: x.shift(1).rolling(w,min_periods=1).mean()
            )
            players[f"win_pct_away_roll{w}"] = players.loc[players["is_home"]==0].groupby("team")["win"].transform(
                lambda x: x.shift(1).rolling(w,min_periods=1).mean()
            )

    return players