# ============================================================
# File: data.py
# Path: nba_analytics_core/data.py
#
# Description:
#   Functions for fetching and processing NBA game data.
#   Includes:
#     - Historical game data retrieval
#     - Today's games retrieval
#     - Team stats aggregation
#     - Matchup feature engineering
#     - CSV export of team stats and matchup features
#
# Author: Your Name
# Created: 2025-12-01
# Updated: 2025-12-01
#
# Notes:
#   - Dependencies: pandas, numpy, nba_api, logging
#   - CSV output paths default to "data/team_stats.csv" and "data/matchup_features.csv"
# ============================================================

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ... existing functions (fetch_historical_games, fetch_today_games, build_team_stats, build_matchup_features) ...

def export_team_stats(games, out_file="data/team_stats.csv"):
    """
    Build team stats from games and export to CSV.
    """
    stats = build_team_stats(games)
    df = pd.DataFrame.from_dict(stats, orient="index").reset_index().rename(columns={"index": "team"})
    df.to_csv(out_file, index=False)
    logger.info(f"✅ Team stats exported to {out_file} ({len(df)} rows)")
    return df

def export_matchup_features(today_games, team_stats, out_file="data/matchup_features.csv"):
    """
    Build matchup features for today's games and export to CSV.
    """
    features = []
    for g in today_games:
        home, away = g["home_team"], g["away_team"]
        feats = build_matchup_features(home, away, team_stats)
        feats.update({"home_team": home, "away_team": away})
        features.append(feats)

    df = pd.DataFrame(features)
    df.to_csv(out_file, index=False)
    logger.info(f"✅ Matchup features exported to {out_file} ({len(df)} rows)")
    return df
