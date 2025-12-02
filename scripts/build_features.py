# ============================================================
# File: build_features.py
# Path: scripts/build_features.py
# Description:
#   Builds NBA game-level features with player stats, team trends,
#   home/away splits, usage rate, opponent strength, rest, injuries, odds.
# ============================================================

import os
import sys
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# ----------------------------
# Config & schema
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
# Feature engineering
# ----------------------------
def _rolling_features(df: pd.DataFrame, group_col: str, date_col: str, cols: List[str],
                      windows: Tuple[int,...]=(3,5,10), shift_by_one: bool=True) -> pd.DataFrame:
    df = df.sort_values([group_col, date_col]).copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    shifted = df.groupby(group_col)[cols].shift(1) if shift_by_one else df[cols]
    for w in windows:
        for c in cols:
            df[f"{c}_roll{w}"] = shifted.groupby(df[group_col])[c].transform(lambda x: x.rolling(w,min_periods=1).mean())
            df[f"{c}_wroll{w}"] = shifted.groupby(df[group_col])[c].transform(
                lambda x: x.rolling(w,min_periods=1).apply(lambda arr: np.average(arr, weights=np.linspace(1,2,len(arr))), raw=True)
            )
    return df

def _team_opponent_strength(games: pd.DataFrame) -> pd.DataFrame:
    home = games.rename(columns={"home_team":"team","away_team":"opponent","away_points":"opp_points","home_points":"points"})[['date','team','opponent','points','opp_points']]
    away = games.rename(columns={"away_team":"team","home_team":"opponent","home_points":"opp_points","away_points":"points"})[['date','team','opponent','points','opp_points']]
    long = pd.concat([home, away], ignore_index=True)
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.sort_values(["team","date"])
    long["points_against_opponent_shift"] = long.groupby("opponent")["points"].shift(1)
    long["opp_def_strength"] = long.groupby("opponent")["points_against_opponent_shift"].transform(lambda s: s.rolling(10,min_periods=3).mean())
    return long[["date","team","opponent","opp_def_strength"]].copy()

def _injury_impact(inj: pd.DataFrame) -> pd.DataFrame:
    if "impact_minutes" not in inj.columns:
        inj["impact_minutes"] = 0
    inj["impact_minutes"] = pd.to_numeric(inj["impact_minutes"], errors="coerce").fillna(0)
    impact = inj.groupby(["date","team"])["impact_minutes"].sum().reset_index().rename(columns={"impact_minutes":"team_impact_minutes"})
    return impact

def _rest_features(games: pd.DataFrame) -> pd.DataFrame:
    def team_schedule(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
        s = pd.DataFrame({"team": df[team_col], "date": pd.to_datetime(df["date"], errors="coerce")})
        s = s.dropna(subset=["date"])
        s = s.groupby("team")["date"].apply(lambda d: d.sort_values()).reset_index()
        s["prev_date"] = s.groupby("team")["date"].shift(1)
        s["rest_days"] = (s["date"] - s["prev_date"]).dt.days.fillna(5)
        s["b2b"] = (s["rest_days"] <= 1).astype(int)
        return s[["team","date","rest_days","b2b"]]
    home = team_schedule(games.rename(columns={"home_team":"team"})["team","date"], "team")
    away = team_schedule(games.rename(columns={"away_team":"team"})["team","date"], "team")
    return pd.concat([home, away], ignore_index=True)

def _moneyline_to_decimal(ml: float) -> Optional[float]:
    if pd.isna(ml):
        return None
    try:
        ml = float(ml)
        if ml > 0: return 1.0 + (ml/100.0)
        elif ml < 0: return 1.0 + (100.0/abs(ml))
        else: return None
    except: return None

# ----------------------------
# Player usage rate
# ----------------------------
def _compute_usage_rate(players: pd.DataFrame) -> pd.DataFrame:
    for col in ["fga","fta","turnovers"]:
        if col not in players.columns: players[col]=0
    players["usage"] = players["fga"] + 0.44*players["fta"] + players["turnovers"]
    team_totals = players.groupby(["team","date"])[["usage"]].sum().rename(columns={"usage":"team_usage"})
    players = players.merge(team_totals, on=["team","date"], how="left")
    players["usage_rate"] = players["usage"] / players["team_usage"].replace(0,np.nan)
    return players

# ----------------------------
# Home/Away splits
# ----------------------------
def _home_away_splits(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    players = players.merge(games[["date","home_team","away_team"]], left_on=["team","date"], right_on=["home_team","date"], how="left")
    players["is_home"] = (players["team"]==players["home_team"]).astype(int)
    agg_cols = ["points","rebounds","assists"]
    windows=(3,5,10)
    for w in windows:
        for col in agg_cols:
            home_feat = players[players["is_home"]==1].groupby("team")[col].transform(lambda x: x.shift(1).rolling(w,min_periods=1).mean())
            away_feat = players[players["is_home"]==0].groupby("team")[col].transform(lambda x: x.shift(1).rolling(w,min_periods=1).mean())
            players[f"{col}_home_roll{w}"] = home_feat
            players[f"{col}_away_roll{w}"] = away_feat
    return players

# ----------------------------
# Main feature builder
# ----------------------------
def build_and_save_features(games_path=os.path.join(DEFAULT_DATA_DIR,"games.csv"),
                            player_stats_path=os.path.join(DEFAULT_DATA_DIR,"player_stats.csv"),
                            injuries_path=os.path.join(DEFAULT_DATA_DIR,"injuries.csv"),
                            odds_path=os.path.join(DEFAULT_DATA_DIR,"odds.csv"),
                            output_path=DEFAULT_FEATURES_PATH) -> pd.DataFrame:
    _ensure_dir(DEFAULT_DATA_DIR)
    
    # Load
    games = _safe_read(games_path)
    players = _safe_read(player_stats_path)
    injuries = _safe_read(injuries_path, required=False)
    odds = _safe_read(odds_path, required=False)

    # Schema
    _ensure_columns(games, REQUIRED_GAMES_COLUMNS, "games.csv")
    _ensure_columns(players, REQUIRED_PLAYER_COLUMNS, "player_stats.csv")

    games = _parse_dates(games, ["date"])
    players = _parse_dates(players, ["date"])
    if injuries is not None: injuries = _parse_dates(injuries, ["date"]).dropna(subset=["date"])

    # Player rolling stats
    base_cols = ["points","rebounds","assists"]
    players = _rolling_features(players,"player","date",base_cols,(3,5,10),True)
    players = _compute_usage_rate(players)
    players = _home_away_splits(players, games)

    # Team aggregation
    agg_cols = {c:"mean" for c in players.columns if c not in ["player","team","date","fga","fta","turnovers","usage","team_usage","is_home"]}
    team_trends = players.groupby(["team","date"]).agg(agg_cols).reset_index()

    # Opponent strength
    opp_strength = _team_opponent_strength(games)

    # Rest
    rest = _rest_features(games)

    # Injuries
    inj_impact = _injury_impact(injuries) if injuries is not None else pd.DataFrame(columns=["date","team","team_impact_minutes"])

    # Merge home/away features
    home = games.merge(team_trends.rename(columns={"team":"home_team"}), on=["home_team","date"], how="left") \
                .merge(opp_strength.rename(columns={"team":"home_team","opponent":"away_team"}), on=["home_team","away_team","date"], how="left") \
                .merge(rest.rename(columns={"team":"home_team"}), on=["home_team","date"], how="left") \
                .merge(inj_impact.rename(columns={"team":"home_team"}), on=["home_team","date"], how="left")
    away = games.merge(team_trends.rename(columns={"team":"away_team"}), on=["away_team","date"], how="left") \
                .merge(opp_strength.rename(columns={"team":"away_team","opponent":"home_team"}), on=["away_team","home_team","date"], how="left") \
                .merge(rest.rename(columns={"team":"away_team"}), on=["away_team","date"], how="left") \
                .merge(inj_impact.rename(columns={"team":"away_team"}), on=["away_team","date"], how="left")

    # Odds
    if odds is not None and len(odds)>0:
        odds["home_decimal_odds"] = odds["home_moneyline"].apply(_moneyline_to_decimal)
        odds["away_decimal_odds"] = odds["away_moneyline"].apply(_moneyline_to_decimal)
        home = home.merge(odds[["game_id","home_decimal_odds","spread","total"]], on="game_id", how="left")
        away = away.merge(odds[["game_id","away_decimal_odds","spread","total"]], on="game_id", how="left")

    # Label
    df = games.copy()
    df["home_win"] = (df["home_points"]>df["away_points"]).astype(int)

    # Merge
    base_keys = ["game_id","date","home_team","away_team"]
    home_feat = home.rename(columns={c:f"home_{c}" for c in home.columns if c not in base_keys})
    away_feat = away.rename(columns={c:f"away_{c}" for c in away.columns if c not in base_keys})
    features = df.merge(home_feat, on=base_keys, how="left").merge(away_feat, on=base_keys, how="left")

    # Defaults
    for k,v in {"home_team_impact_minutes":0,"away_team_impact_minutes":0,"home_rest_days":3,"away_rest_days":3,"home_b2b":0,"away_b2b":0}.items():
        if k in features.columns: features[k]=features[k].fillna(v)

    # Clip
    for col,(low,high) in {"home_minutes":(0,48),"away_minutes":(0,48),"home_rest_days":(0,7),"away_rest_days":(0,7)}.items():
        if col in features.columns: features[col]=features[col].clip(lower=low,upper=high)

    # Save
    features.to_csv(output_path, index=False)
    ts_path = os.path.join(DEFAULT_DATA_DIR,f"training_features_{_timestamp_str()}.csv")
    features.to_csv(ts_path, index=False)
    logger.info(f"✅ Features saved to {output_path}, backup {ts_path}")

    # Metadata
    schema_meta = {"generated_at":datetime.now().isoformat(),"rows":len(features),
                   "columns":features.columns.tolist(),"base_keys":base_keys,
                   "source_files":{"games":games_path,"players":player_stats_path,"injuries":injuries_path,"odds":odds_path}}
    pd.Series(schema_meta).to_json(os.path.join(DEFAULT_DATA_DIR,"training_features_meta.json"))
    logger.info(f"🧾 Feature metadata saved")

    return features

# ----------------------------
# CLI
# ----------------------------
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build NBA training features")
    parser.add_argument("--games", type=str, default=os.path.join(DEFAULT_DATA_DIR,"games.csv"))
    parser.add_argument("--players", type=str, default=os.path.join(DEFAULT_DATA_DIR,"player_stats.csv"))
    parser.add_argument("--injuries", type=str, default=os.path.join(DEFAULT_DATA_DIR,"injuries.csv"))
    parser.add_argument("--odds", type=str, default=os.path.join(DEFAULT_DATA_DIR,"odds.csv"))
    parser.add_argument("--out", type=str, default=DEFAULT_FEATURES_PATH)
    args = parser.parse_args()
    try:
        build_and_save_features(args.games,args.players,args.injuries,args.odds,args.out)
    except Exception as e:
        logger.error(f"Feature building failed: {e}")
        sys.exit(1)
