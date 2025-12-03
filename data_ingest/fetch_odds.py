# ============================================================
# File: data_ingest/fetch_odds.py
# Purpose: Fetch betting odds data (placeholder or API integration)
# ============================================================

import os
import pandas as pd
from core.config import ODDS_SOURCE_URL, BASE_DATA_DIR
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.utils import ensure_columns

logger = setup_logger("fetch_odds")


def fetch_odds(season: str = "2024-25") -> pd.DataFrame:
    """
    Fetch betting odds for a given season.
    Currently uses placeholder data; replace with API integration.
    Saves to BASE_DATA_DIR/{season}_odds.csv.
    """
    try:
        logger.info(f"Fetching odds from {ODDS_SOURCE_URL} (placeholder)...")

        # Placeholder odds data
        df = pd.DataFrame({
            "game_id": [1, 2, 3],
            "home_team": ["X", "Y", "Z"],
            "away_team": ["A", "B", "C"],
            "decimal_odds": [1.8, 2.0, 1.5],
        })

        # Validate required columns
        ensure_columns(df, {"game_id", "home_team", "away_team", "decimal_odds"}, "odds")

        out_file = os.path.join(BASE_DATA_DIR, f"{season}_odds.csv")
        os.makedirs(BASE_DATA_DIR, exist_ok=True)
        df.to_csv(out_file, index=False)

        logger.info(f"✅ Odds saved to {out_file} ({len(df)} rows)")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch odds: {e}")
        raise PipelineError(f"Odds fetching failed: {e}")