# ============================================================
# File: scripts/fetch_game_results.py
# Purpose: Fetch NBA game results (synthetic or real) and save to CSV
# ============================================================

import argparse
import pandas as pd
from pathlib import Path
from core.config import GAME_RESULTS_FILE, ensure_dirs
from core.log_config import setup_logger
from core.exceptions import DataError

logger = setup_logger("fetch_game_results")


def fetch_synthetic_results() -> pd.DataFrame:
    """Generate a small synthetic dataset of game results for testing."""
    data = [
        {"game_id": 1, "date": "2025-11-25", "home_team": "DEN", "away_team": "LAL",
         "home_score": 110, "away_score": 102, "home_win": 1},
        {"game_id": 2, "date": "2025-11-25", "home_team": "BOS", "away_team": "PHI",
         "home_score": 95, "away_score": 100, "home_win": 0},
        {"game_id": 3, "date": "2025-11-26", "home_team": "MIA", "away_team": "NYK",
         "home_score": 105, "away_score": 99, "home_win": 1},
        {"game_id": 4, "date": "2025-11-26", "home_team": "GSW", "away_team": "SAS",
         "home_score": 120, "away_score": 115, "home_win": 1},
    ]
    return pd.DataFrame(data)


def fetch_real_results() -> pd.DataFrame:
    """
    Placeholder for real API fetching logic.
    Replace with code that calls your NBA stats API or database.
    """
    raise DataError("Real game results fetching not implemented. Use --use_synthetic for testing.")


def main(use_synthetic: bool = False):
    try:
        ensure_dirs()

        if use_synthetic:
            logger.info("Fetching synthetic game results...")
            df = fetch_synthetic_results()
        else:
            logger.info("Fetching real game results...")
            df = fetch_real_results()

        df.to_csv(GAME_RESULTS_FILE, index=False)
        logger.info(f"Game results saved to {GAME_RESULTS_FILE}")

    except Exception as e:
        logger.error(f"Failed to fetch game results: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA game results")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data instead of real API")
    args = parser.parse_args()

    main(use_synthetic=args.use_synthetic)