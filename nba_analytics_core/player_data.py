# File: nba_analytics_core/player_data.py

import pandas as pd
import logging
from nba_api.stats.endpoints import leaguedashplayerstats

logging.basicConfig(level=logging.INFO)

def fetch_player_season_stats(season: str = "2025-26", per_mode: str = "PerGame") -> pd.DataFrame:
    """
    Fetches real player season statistics using nba_api.
    Returns DataFrame with PTS, REB, AST, TS%, etc.
    """
    logging.info(f"Fetching {per_mode} player stats for season {season}...")

    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense="Base"
        )
        df = stats.get_data_frames()[0]
    except Exception as e:
        logging.error(f"Failed to fetch player stats: {e}")
        return pd.DataFrame()

    # Select key columns for dashboard
    cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "REB", "AST", "TS_PCT"]
    df = df[[c for c in cols if c in df.columns]]

    logging.info(f"✔ Retrieved stats for {len(df)} players.")
    return df.sort_values(by=["PTS", "AST", "REB"], ascending=False).reset_index(drop=True)


def build_player_leaderboards(df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Builds a dictionary of DataFrames representing different leaderboards.
    """
    if df.empty:
        return {}

    leaderboards = {
        "Scoring (PTS)": df.nlargest(top_n, "PTS"),
        "Rebounding (REB)": df.nlargest(top_n, "REB"),
        "Assists (AST)": df.nlargest(top_n, "AST"),
        "Efficiency (TS_PCT)": df.nlargest(top_n, "TS_PCT"),
        "Minutes (MIN)": df.nlargest(top_n, "MIN"),
        "Composite Impact": df.assign(IMPACT=df["PTS"] + df["REB"] + df["AST"]).nlargest(top_n, "IMPACT")
    }
    return leaderboards