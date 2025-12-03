# ============================================================
# File: scripts/fetch_new_games.py
# Purpose: Fetch today's games or generate synthetic games for CI/testing
# ============================================================

import os
import pandas as pd
from core.config import NEW_GAMES_FILE, BASE_DATA_DIR
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.utils import ensure_columns

logger = setup_logger("fetch_new_games")


def main(use_synthetic: bool = False) -> None:
    """
    Fetch today's games. If use_synthetic=True, generate synthetic games for CI/testing.
    Saves to NEW_GAMES_FILE with raw stats (feature engineering added later).
    """
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    if use_synthetic:
        logger.info("⚙️ Using synthetic new_games.csv for CI/testing...")
        df = pd.DataFrame({
            "TEAM_HOME": ["SYN_A", "SYN_B"],
            "TEAM_AWAY": ["SYN_C", "SYN_D"],
            "AGE": [25, 28],
            "PTS": [15, 20],
            "AST": [5, 7],
            "REB": [4, 6],
            "GAMES_PLAYED": [10, 12],
            "decimal_odds": [1.8, 2.1],
        })
        df.to_csv(NEW_GAMES_FILE, index=False)
        logger.info(f"✅ Synthetic new_games.csv saved to {NEW_GAMES_FILE}")
        return

    # --- Real scraping logic placeholder ---
    try:
        logger.info("Fetching real games...")
        df = pd.DataFrame({
            "TEAM_HOME": ["LAL", "GSW"],
            "TEAM_AWAY": ["BOS", "MIA"],
            "AGE": [30, 27],
            "PTS": [25, 29],
            "AST": [7, 6],
            "REB": [8, 5],
            "GAMES_PLAYED": [20, 18],
            "decimal_odds": [1.9, 2.2],
        })

        # Validate required columns
        ensure_columns(df, {"TEAM_HOME", "TEAM_AWAY", "PTS", "AST", "REB", "GAMES_PLAYED"}, "new games")

        df.to_csv(NEW_GAMES_FILE, index=False)
        logger.info(f"✅ Real new_games.csv saved to {NEW_GAMES_FILE}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch real games: {e}")
        logger.info("Falling back to synthetic games...")
        main(use_synthetic=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch today's games")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Generate synthetic games instead of scraping")
    args = parser.parse_args()
    main(use_synthetic=args.use_synthetic)