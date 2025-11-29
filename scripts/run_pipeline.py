import os
import yaml
import logging
import sqlite3
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder
from utils.notify import send_daily_picks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
MODEL_PATH = os.path.join("models", CONFIG["model"]["filename"])
model = joblib.load(MODEL_PATH)

# ---------------------------
# 1. Fetch NBA Games
# ---------------------------
def season_to_str(year: int) -> str:
    return f"{year}-{str(year+1)[-2:]}"

def fetch_nba_games(team_id=None, season=2024):
    try:
        season_str = season_to_str(season)
        gamefinder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id,
            season_nullable=season_str
        )
        df = gamefinder.get_data_frames()[0]
        logging.info(f"✅ Fetched {len(df)} games for season {season_str}")
        return df
    except Exception as e:
        logging.error(f"❌ Failed to fetch games: {e}")
        return pd.DataFrame()

def store_games(df):
    if df.empty:
        return
    required_cols = ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PTS", "REB", "AST", "WL", "MATCHUP"]
    df_clean = df[required_cols].drop_duplicates(subset=["GAME_ID", "TEAM_ABBREVIATION"])
    with sqlite3.connect(DB_PATH) as con:
        df_clean.to_sql("nba_games", con, if_exists="append", index=False)
    logging.info(f"✔ Stored {len(df_clean)} games")

# ---------------------------
# 2. Fetch Odds
# ---------------------------
def fetch_odds():
    try:
        url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
        params = {
            "apiKey": CONFIG["odds"]["api_key"],
            "regions": CONFIG["odds"]["regions"],
            "markets": CONFIG["odds"]["markets"],
            "oddsFormat": CONFIG["odds"]["odds_format"],
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        logging.info("✅ Odds data fetched")
        return resp.json()
    except Exception as e:
        logging.error(f"❌ Failed to fetch odds: {e}")
        return []

# ---------------------------
# 3. Feature Engineering
# ---------------------------
def build_features(df, team, opponent):
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    for stat in ["PTS", "REB", "AST"]:
        df[f"{stat}_avg"] = df.groupby("TEAM_ABBREVIATION")[stat].transform(
            lambda x: x.rolling(CONFIG["features"]["rolling_games"], min_periods=1).mean()
        )
    df_team = df[df["TEAM_ABBREVIATION"] == team].tail(1)
    df_opp = df[df["TEAM_ABBREVIATION"] == opponent].tail(1)
    if df_team.empty or df_opp.empty:
        return None
    row, opp_row = df_team.iloc[0], df_opp.iloc[0]
    feats = np.array([[row["PTS_avg"], row["REB_avg"], row["AST_avg"],
                       opp_row["PTS_avg"], opp_row["REB_avg"], opp_row["AST_avg"]]])
    if np.isnan(feats).any():
        return None
    return feats

# ---------------------------
# 4. EV & Kelly
# ---------------------------
def calculate_ev(prob, odds):
    return (prob * odds) - 1

def kelly_stake(prob, odds, bankroll):
    b = odds - 1
    q = 1 - prob
    f = (b * prob - q) / b if b != 0 else 0
    return max(0, f) * bankroll

# ---------------------------
# 5. Merge Games + Odds → Predictions
# ---------------------------
def generate_picks():
    with sqlite3.connect(DB_PATH) as con:
        db_df = pd.read_sql("SELECT * FROM nba_games", con)

    odds_json = fetch_odds()
    if not odds_json:
        return pd.DataFrame()

    picks = []
    for game in odds_json:
        if not game.get("bookmakers"):
            continue
        try:
            home = game["home_team"]
            away = game["away_team"]
            outcomes = game["bookmakers"][0]["markets"][0]["outcomes"]
