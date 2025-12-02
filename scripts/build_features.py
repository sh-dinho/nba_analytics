# File: scripts/build_features.py
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

# ----------------------------
# Configuration & schema
# ----------------------------
REQUIRED_GAMES_COLUMNS = {
    "game_id", "date", "home_team", "away_team", "home_points", "away_points"
}
REQUIRED_PLAYER_COLUMNS = {
    "date", "player", "team", "points", "rebounds", "assists", "minutes"
}
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
# Feature engineering helpers
# ----------------------------
def _rolling_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    cols: List[str],
    windows: Tuple[int, ...] = (3, 5, 10),
    shift_by_one: bool = True
) -> pd.DataFrame:
    """
    Compute rolling means and weighted rolling means per group ordered by date.
    shift_by_one=True ensures we only use data strictly before the current date
    to avoid leakage (i.e., today's performance isn't in today's features).
    """
    df = df.sort_values([group_col, date_col]).copy()

    # Ensuring numeric
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional shift to avoid leakage
    shifted = df.groupby(group_col)[cols].shift(1) if shift_by_one else df[cols]
    for w in windows:
        for c in cols:
            df[f"{c}_roll{w}"] = (
                shifted.groupby(df[group_col])[c]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
        for c in cols:
            # Weighted rolling (linearly increasing weights)
            df[f"{c}_wroll{w}"] = (
                shifted.groupby(df[group_col])[c]
                .transform(
                    lambda x: x.rolling(w, min_periods=1)
                    .apply(lambda arr: np.average(arr, weights=np.linspace(1, 2, len(arr))), raw=True)
                )
            )
    return df


def _team_opponent_strength(games: pd.DataFrame) -> pd.DataFrame:
    """
    Builds long format of team-game results and computes opponent defensive strength
    as rolling mean of points conceded over last N games. Uses shift to avoid leakage.
    """
    # Home perspective
    home = games.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "away_points": "opp_points",
            "home_points": "points",
        }
    )[['date', 'team', 'opponent', 'points', 'opp_points']]

    # Away perspective
    away = games.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "home_points": "opp_points",
            "away_points": "points",
        }
    )[['date', 'team', 'opponent', 'points', 'opp_points']]

    long = pd.concat([home, away], ignore_index=True).copy()
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.sort_values(["team", "date"])

    # Opponent defensive strength: average points scored against opponent
    long["points_against_opponent"] = long["points"]

    # To avoid leakage, compute opponent history shifted by one
    long["points_against_opponent_shift"] = long.groupby("opponent")["points_against_opponent"].shift(1)

    long["opp_def_strength"] = (
        long.groupby("opponent")["points_against_opponent_shift"]
        .transform(lambda s: s.rolling(10, min_periods=3).mean())
    )

    return long[["date", "team", "opponent", "opp_def_strength"]].copy()


def _injury_impact(inj: pd.DataFrame) -> pd.DataFrame:
    if "impact_minutes" not in inj.columns:
        inj["impact_minutes"] = 0
    inj["impact_minutes"] = pd.to_numeric(inj["impact_minutes"], errors="coerce").fillna(0)
    impact = (
        inj.groupby(["date", "team"])["impact_minutes"]
        .sum()
        .reset_index()
        .rename(columns={"impact_minutes": "team_impact_minutes"})
    )
    return impact


def _rest_features(games: pd.DataFrame) -> pd.DataFrame:
    def team_schedule(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
        s = pd.DataFrame({"team": df[team_col], "date": pd.to_datetime(df["date"], errors="coerce")})
        s = s.dropna(subset=["date"])
        s = s.groupby("team")["date"].apply(lambda d: d.sort_values()).reset_index()
        s["prev_date"] = s.groupby("team")["date"].shift(1)
        s["rest_days"] = (s["date"] - s["prev_date"]).dt.days
        s["rest_days"] = s["rest_days"].fillna(5)
        s["b2b"] = (s["rest_days"] <= 1).astype(int)
        return s[["team", "date", "rest_days", "b2b"]]

    home = team_schedule(games[["date", "home_team"]].rename(columns={"home_team": "team"}), "team")
    away = team_schedule(games[["date", "away_team"]].rename(columns={"away_team": "team"}), "team")
    rest = pd.concat([home, away], ignore_index=True)
    return rest


def _moneyline_to_decimal(ml: float) -> Optional[float]:
    """
    Convert American moneyline to decimal odds:
    +150 -> 2.50 ; -200 -> 1.50
    Returns None if ml is NaN or invalid.
    """
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
    except Exception:
        return None


# ----------------------------
# Main feature builder
# ----------------------------
def build_and_save_features(
    games_path: str = os.path.join(DEFAULT_DATA_DIR, "games.csv"),
    player_stats_path: str = os.path.join(DEFAULT_DATA_DIR, "player_stats.csv"),
    injuries_path: str = os.path.join(DEFAULT_DATA_DIR, "injuries.csv"),
    odds_path: str = os.path.join(DEFAULT_DATA_DIR, "odds.csv"),
    output_path: str = DEFAULT_FEATURES_PATH
) -> pd.DataFrame:
    _ensure_dir("features")
    _ensure_dir(DEFAULT_DATA_DIR)

    # Load
    try:
        games = _safe_read(games_path, required=True)
    except FileNotFoundError as e:
        logger.error(f"{e}")
        return pd.DataFrame()

    players = _safe_read(player_stats_path, required=True)
    injuries = _safe_read(injuries_path, required=False)
    odds = _safe_read(odds_path, required=False)

    # Schema checks
    _ensure_columns(games, REQUIRED_GAMES_COLUMNS, os.path.basename(games_path))
    _ensure_columns(players, REQUIRED_PLAYER_COLUMNS, os.path.basename(player_stats_path))
    if injuries is not None and len(injuries) > 0:
        missing_inj = [c for c in OPTIONAL_INJURY_COLUMNS if c not in injuries.columns]
        if missing_inj:
            logger.warning(f"{os.path.basename(injuries_path)} missing columns {missing_inj}. Defaulting impact_minutes=0.")

    # Dates
    games = _parse_dates(games, ["date"])
    players = _parse_dates(players, ["date"])
    if injuries is not None and len(injuries) > 0:
        injuries = _parse_dates(injuries, ["date"])
        injuries = injuries.dropna(subset=["date"])

    # Player rolling trends (leakage-safe via shift)
    base_cols = ["points", "rebounds", "assists"]
    players = _rolling_features(players, "player", "date", base_cols, windows=(3, 5, 10), shift_by_one=True)

    # Team-level aggregation of player trends
    agg_cols = {
        "points_roll3": "mean",
        "rebounds_roll3": "mean",
        "assists_roll3": "mean",
        "points_wroll5": "mean",
        "rebounds_wroll5": "mean",
        "assists_wroll5": "mean",
        "points_wroll10": "mean",
        "rebounds_wroll10": "mean",
        "assists_wroll10": "mean",
        "minutes": "mean",
    }
    team_trends = players.groupby(["team", "date"]).agg(agg_cols).reset_index()

    # Opponent defensive strength (leakage-safe via shift)
    opp_strength = _team_opponent_strength(games)
    opp_strength["date"] = pd.to_datetime(opp_strength["date"], errors="coerce")

    # Rest features
    rest = _rest_features(games)
    rest["date"] = pd.to_datetime(rest["date"], errors="coerce")

    # Injuries impact
    if injuries is not None and len(injuries) > 0:
        inj_impact = _injury_impact(injuries)
        inj_impact["date"] = pd.to_datetime(inj_impact["date"], errors="coerce")
    else:
        inj_impact = pd.DataFrame(columns=["date", "team", "team_impact_minutes"])

    # Home features
    home = (
        games.merge(team_trends.rename(columns={"team": "home_team"}), on=["home_team", "date"], how="left")
             .merge(
                 opp_strength.rename(columns={"team": "home_team", "opponent": "away_team"}),
                 on=["home_team", "away_team", "date"], how="left"
             )
             .merge(rest.rename(columns={"team": "home_team"}), on=["home_team", "date"], how="left")
             .merge(inj_impact.rename(columns={"team": "home_team"}), on=["home_team", "date"], how="left")
    )

    # Away features
    away = (
        games.merge(team_trends.rename(columns={"team": "away_team"}), on=["away_team", "date"], how="left")
             .merge(
                 opp_strength.rename(columns={"team": "away_team", "opponent": "home_team"}),
                 on=["away_team", "home_team", "date"], how="left"
             )
             .merge(rest.rename(columns={"team": "away_team"}), on=["away_team", "date"], how="left")
             .merge(inj_impact.rename(columns={"team": "away_team"}), on=["away_team", "date"], how="left")
    )

    # Odds join (optional)
    if odds is not None and len(odds) > 0:
        # Try to ensure expected columns
        missing_odds = [c for c in OPTIONAL_ODDS_COLUMNS if c not in odds.columns]
        if missing_odds:
            logger.warning(f"odds.csv is missing columns: {missing_odds}. Skipping odds merge.")
        else:
            # Convert moneyline to decimal to standardize downstream usage
            odds["home_decimal_odds"] = odds["home_moneyline"].apply(_moneyline_to_decimal)
            odds["away_decimal_odds"] = odds["away_moneyline"].apply(_moneyline_to_decimal)
            home = home.merge(
                odds[["game_id", "home_decimal_odds", "spread", "total"]],
                on="game_id", how="left"
            )
            away = away.merge(
                odds[["game_id", "away_decimal_odds", "spread", "total"]],
                on="game_id", how="left"
            )

    # Label
    df = games.copy()
    df["home_win"] = (df["home_points"] > df["away_points"]).astype(int)

    # Merge keys
    base_keys = ["game_id", "date", "home_team", "away_team"]

    # Prefix non-key columns
    home_feat = home.rename(columns={c: f"home_{c}" for c in home.columns if c not in base_keys})
    away_feat = away.rename(columns={c: f"away_{c}" for c in away.columns if c not in base_keys})

    # Merge back to game-level
    features = (
        df.merge(home_feat, on=base_keys, how="left")
          .merge(away_feat, on=base_keys, how="left")
    )

    # Defaults and NA handling
    fill_defaults = {
        "home_team_impact_minutes": 0,
        "away_team_impact_minutes": 0,
        "home_rest_days": 3,
        "away_rest_days": 3,
        "home_b2b": 0,
        "away_b2b": 0,
    }
    for k, v in fill_defaults.items():
        if k in features.columns:
            features[k] = features[k].fillna(v)

    # Final sanity: ensure no future leakage columns inadvertently merged
    # (E.g., opp_def_strength could be NaN early season; keep NaNs rather than backfill.)
    # Optional: cap extreme values to reduce outlier effects
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    # Example clipping for stability
    clip_map = {
        "home_minutes": (0, 48), "away_minutes": (0, 48),
        "home_rest_days": (0, 7), "away_rest_days": (0, 7),
    }
    for col, (low, high) in clip_map.items():
        if col in features.columns:
            features[col] = features[col].clip(lower=low, upper=high)

    # Save with timestamped backup
    _ensure_dir(DEFAULT_DATA_DIR)
    features.to_csv(output_path, index=False)
    ts_path = os.path.join(DEFAULT_DATA_DIR, f"training_features_{_timestamp_str()}.csv")
    features.to_csv(ts_path, index=False)

    logger.info(f"✅ Training features saved to {output_path}")
    logger.info(f"📦 Timestamped backup saved to {ts_path}")
    logger.info(f"Feature rows: {len(features)}, columns: {len(features.columns)}")

    # Optional: export feature metadata (schema) for downstream validation
    schema_meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(features),
        "columns": features.columns.tolist(),
        "base_keys": base_keys,
        "source_files": {
            "games": games_path,
            "players": player_stats_path,
            "injuries": injuries_path,
            "odds": odds_path,
        }
    }
    meta_path = os.path.join(DEFAULT_DATA_DIR, "training_features_meta.json")
    pd.Series(schema_meta).to_json(meta_path)
    logger.info(f"🧾 Feature metadata saved to {meta_path}")

    return features


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build training features from raw NBA datasets")
    parser.add_argument("--games", type=str, default=os.path.join(DEFAULT_DATA_DIR, "games.csv"))
    parser.add_argument("--players", type=str, default=os.path.join(DEFAULT_DATA_DIR, "player_stats.csv"))
    parser.add_argument("--injuries", type=str, default=os.path.join(DEFAULT_DATA_DIR, "injuries.csv"))
    parser.add_argument("--odds", type=str, default=os.path.join(DEFAULT_DATA_DIR, "odds.csv"))
    parser.add_argument("--out", type=str, default=DEFAULT_FEATURES_PATH)

    args = parser.parse_args()

    try:
        build_and_save_features(
            games_path=args.games,
            player_stats_path=args.players,
            injuries_path=args.injuries,
            odds_path=args.odds,
            output_path=args.out
        )
    except Exception as e:
        logger.error(f"Feature building failed: {e}")
        sys.exit(1)