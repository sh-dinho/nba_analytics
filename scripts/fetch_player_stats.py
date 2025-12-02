# File: scripts/fetch_player_stats.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def fetch_stats_bball_ref(season_year: str) -> pd.DataFrame:
    """
    Fetch player stats from Basketball Reference and normalize column names.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_year}_per_game.html"
    logger.info(f"Fetching player stats from {url}")

    # Read the first table
    tables = pd.read_html(url)
    df = tables[0]

    # Drop header rows that repeat
    df = df[df["Player"] != "Player"]

    # Normalize column names
    rename_map = {
        "Player": "PLAYER_NAME",
        "Pos": "POSITION",
        "Age": "AGE",
        "Tm": "TEAM_ABBREVIATION",
        "G": "GAMES_PLAYED",
        "GS": "GAMES_STARTED",
        "MP": "MINUTES",
        "FG": "FGM",
        "FGA": "FGA",
        "FG%": "FG_PCT",
        "3P": "FG3M",
        "3PA": "FG3A",
        "3P%": "FG3_PCT",
        "FT": "FTM",
        "FTA": "FTA",
        "FT%": "FT_PCT",
        "ORB": "OREB",
        "DRB": "DREB",
        "TRB": "REB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TO",
        "PF": "PF",
        "PTS": "PTS"
    }
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required_cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Available: {df.columns.tolist()}")

    return df[required_cols]