# Path: nba_analytics_core/player_data.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def fetch_player_season_stats(season: str = "2025-26", per_mode: str = "PerGame") -> pd.DataFrame:
    """
    MOCK: Fetches player season statistics (PTS, REB, AST, etc.) for prototyping dashboards.
    Replace with nba_api calls for production.
    """
    logging.info(f"MOCK: Fetching {per_mode} player stats for season {season}...")
    player_names = [
        "LeBron James","Stephen Curry","Nikola Jokic","Jayson Tatum",
        "Luka Doncic","Victor Wembanyama","Chet Holmgren","Shaedon Sharpe",
        "Jalen Williams","Alperen Sengun","Cooper Flagg","Tyrese Haliburton",
        "Zion Williamson","Anthony Edwards","Ja Morant","Kawhi Leonard"
    ]
    teams = {
        "LeBron James":"LAL","Stephen Curry":"GSW","Nikola Jokic":"DEN","Jayson Tatum":"BOS",
        "Luka Doncic":"DAL","Victor Wembanyama":"SAS","Chet Holmgren":"OKC","Shaedon Sharpe":"POR",
        "Jalen Williams":"OKC","Alperen Sengun":"HOU","Cooper Flagg":"MIA","Tyrese Haliburton":"IND",
        "Zion Williamson":"NOP","Anthony Edwards":"MIN","Ja Morant":"MEM","Kawhi Leonard":"LAC"
    }
    data = []
    for name in player_names:
        is_star = name in ["LeBron James","Nikola Jokic","Luka Doncic"]
        pts_base = 20 if is_star else 15
        reb_base = 7 if is_star else 5
        ast_base = 6 if is_star else 4
        pts = pts_base + np.random.rand() * 10
        reb = reb_base + np.random.rand() * 5
        ast = ast_base + np.random.rand() * 5
        ts_pct = 0.550 + np.random.rand() * 0.15
        data.append({
            "PLAYER_NAME": name,
            "TEAM_ABBREVIATION": teams.get(name, 'TBD'),
            "SEASON_ID": season,
            "GP": 70 + np.random.randint(0, 10),
            "MIN": 30.0 + np.random.rand() * 5,
            "PTS": pts if per_mode == "PerGame" else pts * 75,
            "REB": reb if per_mode == "PerGame" else reb * 75,
            "AST": ast if per_mode == "PerGame" else ast * 75,
            "TS_PCT": ts_pct,
        })
    df = pd.DataFrame(data)
    logging.info(f"✔ Generated stats for {len(df)} mock players.")
    return df.sort_values(by=["PTS","AST","REB"], ascending=False).reset_index(drop=True)

def build_player_leaderboards(df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Builds a dictionary of DataFrames representing different leaderboards.
    """
    if df.empty:
        return {}
    leaderboards = {
        "Scoring (PTS)": df.sort_values("PTS", ascending=False).head(top_n),
        "Rebounding (REB)": df.sort_values("REB", ascending=False).head(top_n),
        "Assists (AST)": df.sort_values("AST", ascending=False).head(top_n),
        "Efficiency (TS_PCT)": df.sort_values("TS_PCT", ascending=False).head(top_n)
    }
    return leaderboards