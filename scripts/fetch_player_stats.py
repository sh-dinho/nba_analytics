# ============================================================
# File: scripts/fetch_player_stats.py
# Purpose: Fetch player stats or generate synthetic data for CI/testing
# ============================================================

import os
import pandas as pd
from core.config import PLAYER_STATS_FILE, BASE_DATA_DIR
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.utils import ensure_columns

logger = setup_logger("fetch_player_stats")


def main(use_synthetic: bool = False) -> None:
    """
    Fetch player stats. If use_synthetic=True, generate synthetic data
    for CI/CD or testing environments.
    Saves to PLAYER_STATS_FILE.
    """
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    if use_synthetic:
        logger.info("⚙️ Using synthetic player stats for CI/testing...")
        df = pd.DataFrame({
            "PLAYER_NAME": ["Synthetic Player A", "Synthetic Player B"],
            "TEAM_ABBREVIATION": ["SYN", "SYN"],
            "AGE": [25, 28],
            "POSITION": ["G", "F"],
            "GAMES_PLAYED": [10, 12],
            "PTS": [15, 20],
            "AST": [5, 7],
            "REB": [4, 6],
        })

        # Validate required columns
        ensure_columns(df, {"PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB"}, "synthetic player stats")

        df.to_csv(PLAYER_STATS_FILE, index=False)
        logger.info(f"✅ Synthetic player stats saved to {PLAYER_STATS_FILE}")
        return

    # --- Real scraping logic (example placeholder) ---
    try:
        logger.info("Fetching real player stats...")
        # Replace with actual scraping or API call
        df = pd.DataFrame({
            "PLAYER_NAME": ["LeBron James", "Stephen Curry"],
            "TEAM_ABBREVIATION": ["LAL", "GSW"],
            "AGE": [40, 37],
            "POSITION": ["F", "G"],
            "GAMES_PLAYED": [20, 18],
            "PTS": [25.3, 29.1],
            "AST": [7.2, 6.5],
            "REB": [8.1, 5.2],
        })

        ensure_columns(df, {"PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB"}, "real player stats")

        df.to_csv(PLAYER_STATS_FILE, index=False)
        logger.info(f"✅ Real player stats saved to {PLAYER_STATS_FILE}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch real stats: {e}")
        logger.info("Falling back to synthetic data...")
        main(use_synthetic=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch player stats")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Generate synthetic stats instead of scraping")
    args = parser.parse_args()
    main(use_synthetic=args.use_synthetic)