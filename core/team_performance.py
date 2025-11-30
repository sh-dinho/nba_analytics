import sqlite3
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO)

# --- Load config safely ---
try:
    with open("config.yaml") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    logging.warning("⚠️ config.yaml not found, using defaults.")
    CONFIG = {"database": {"path": "bets.db"}}

DB_PATH = CONFIG["database"]["path"]

def connect():
    """Connect to SQLite database with row factory."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def calculate_team_performance(season: int = None) -> pd.DataFrame:
    """
    Calculate team performance stats from nba_games table.
    Returns DataFrame with team_name, wins, losses, ties, points_scored, points_allowed, win_percentage.
    """
    query = "SELECT * FROM nba_games"
    params = ()
    if season:
        query += " WHERE season=?"
        params = (season,)

    with connect() as con:
        df = pd.read_sql(query, con, params=params)

    if df.empty:
        logging.warning("No games found for performance calculation.")
        return pd.DataFrame()

    teams = set(df["home_team"]).union(set(df["away_team"]))
    records = []

    for team in teams:
        home_games = df[df["home_team"] == team]
        away_games = df[df["away_team"] == team]

        wins = ((home_games["winner"] == team).sum() +
                (away_games["winner"] == team).sum())
        losses = ((home_games["winner"] != team).sum() +
                  (away_games["winner"] != team).sum())
        ties = 0  # NBA games don’t tie

        points_scored = home_games["home_score"].sum() + away_games["away_score"].sum()
        points_allowed = home_games["away_score"].sum() + away_games["home_score"].sum()

        total_games = wins + losses + ties
        win_percentage = wins / total_games if total_games > 0 else 0.0

        records.append({
            "team_name": team,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "points_scored": points_scored,
            "points_allowed": points_allowed,
            "win_percentage": win_percentage,
            "season": season if season else "All"
        })

    return pd.DataFrame(records)

def store_team_performance(season: int = None):
    """
    Store team performance stats into team_performance table.
    """
    df_perf = calculate_team_performance(season)
    if df_perf.empty:
        return

    with connect() as con:
        cur = con.cursor()
        for _, row in df_perf.iterrows():
            cur.execute("""
                INSERT OR REPLACE INTO team_performance
                (team_name, conference, wins, losses, ties, points_scored, points_allowed, win_percentage, season)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["team_name"],
                "Unknown",  # Conference info not available in DB yet
                row["wins"],
                row["losses"],
                row["ties"],
                row["points_scored"],
                row["points_allowed"],
                row["win_percentage"],
                row["season"] if season else 0
            ))
        con.commit()
    logging.info("✔ Team performance stored in DB")

def get_team_performance(season: int = None) -> pd.DataFrame:
    """
    Fetch team performance stats from DB.
    """
    query = "SELECT * FROM team_performance"
    params = ()
    if season:
        query += " WHERE season=?"
        params = (season,)

    with connect() as con:
        df = pd.read_sql(query, con, params=params)
    return df