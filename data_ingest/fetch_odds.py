# data_ingest/fetch_odds.py
import pandas as pd
from core.config import ODDS_SOURCE_URL, DATA_DIR
from core.logging import setup_logger
import os

logger = setup_logger("fetch_odds")

def fetch_odds(season="2024-25"):
    logger.info(f"Fetching odds from {ODDS_SOURCE_URL} (placeholder)...")
    df = pd.DataFrame({
        "game_id": [1,2,3],
        "home_team": ["X","Y","Z"],
        "away_team": ["A","B","C"],
        "decimal_odds": [1.8,2.0,1.5]
    })
    out_file = os.path.join(DATA_DIR, f"{season}_odds.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(out_file, index=False)
    logger.info(f"Odds saved to {out_file}")
    return df
