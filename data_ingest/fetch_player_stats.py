# data_ingest/fetch_player_stats.py
import pandas as pd
from core.config import PLAYER_STATS_URL, DATA_DIR
import os
from core.log_config import setup_logger

logger = setup_logger("fetch_player_stats")

def fetch_player_stats(season="2024-25", resume=True):
    out_file = os.path.join(DATA_DIR, f"{season}_player_stats.csv")
    if resume and os.path.exists(out_file):
        logger.info(f"Resuming from cached stats: {out_file}")
        return pd.read_csv(out_file)
    
    logger.info(f"Fetching player stats from {PLAYER_STATS_URL} (placeholder)...")
    # Placeholder: simulate fetch
    df = pd.DataFrame({
        "player_id": [1,2,3],
        "player_name": ["A","B","C"],
        "team": ["X","Y","Z"],
        "pts": [10,15,20],
        "ast": [5,3,7],
        "reb": [4,6,5],
    })
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(out_file, index=False)
    logger.info(f"Player stats saved to {out_file}")
    return df
