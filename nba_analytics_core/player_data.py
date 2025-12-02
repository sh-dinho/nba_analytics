# Path: nba_analytics_core/player_data.py

import pandas as pd
import logging
from datetime import datetime
from nba_api.stats.endpoints import leaguedashplayerstats

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REQUIRED_COLUMNS = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "REB", "AST", "TS_PCT"]

def fetch_player_season_stats(season: str = "2025-26", per_mode: str = "PerGame") -> pd.DataFrame:
    """
    Fetches real player season statistics using nba_api.
    Returns DataFrame with PTS, REB, AST, TS%, etc.
    """
    logger.info(f"Fetching {per_mode} player stats for season {season}...")

    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense="Base"
        )
        df = stats.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Failed to fetch player stats: {e}")
        return pd.DataFrame()

    # Select key columns for dashboard
    available_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if not available_cols:
        logger.warning("No required columns found in API response.")
        return pd.DataFrame()

    df = df[available_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    logger.info(f"✔ Retrieved stats for {len(df)} players.")
    return df.sort_values(by=["PTS", "AST", "REB"], ascending=False).reset_index(drop=True)


def build_player_leaderboards(df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Builds a dictionary of DataFrames representing different leaderboards.
    """
    if df.empty:
        logger.warning("Cannot build leaderboards: input DataFrame is empty.")
        return {}

    leaderboards = {
        "Scoring (PTS)": df.nlargest(top_n, "PTS"),
        "Rebounding (REB)": df.nlargest(top_n, "REB"),
        "Assists (AST)": df.nlargest(top_n, "AST"),
        "Efficiency (TS_PCT)": df.nlargest(top_n, "TS_PCT"),
        "Minutes (MIN)": df.nlargest(top_n, "MIN"),
        "Composite Impact": df.assign(IMPACT=df["PTS"] + df["REB"] + df["AST"]).nlargest(top_n, "IMPACT")
    }

    logger.info(f"Leaderboards built for top {top_n} players.")
    return leaderboards


def export_leaderboards(leaderboards: dict, out_dir: str = "results", season: str = "2025-26"):
    """
    Export leaderboards to CSV files.
    """
    if not leaderboards:
        logger.warning("No leaderboards to export.")
        return

    import os
    os.makedirs(out_dir, exist_ok=True)

    for name, df in leaderboards.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        file_path = os.path.join(out_dir, f"leaderboard_{safe_name}.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"📊 Leaderboard '{name}' exported to {file_path}")

    # Save metadata
    meta = {
        "season": season,
        "generated_at": datetime.now().isoformat(),
        "leaderboards": list(leaderboards.keys())
    }
    meta_file = os.path.join(out_dir, "leaderboards_meta.json")
    import json
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")