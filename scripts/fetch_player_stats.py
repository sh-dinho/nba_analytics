# ============================================================
# File: scripts/fetch_player_stats.py
# Purpose: Fetch player stats or generate synthetic data for CI/testing
# ============================================================

import os
import pandas as pd
from core.config import PLAYER_STATS_FILE, BASE_DATA_DIR
from core.log_config import setup_logger
from core.exceptions import DataError
from core.utils import ensure_columns

logger = setup_logger("fetch_player_stats")


def _synthetic_stats() -> pd.DataFrame:
    """Generate synthetic player stats for CI/testing."""
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
    ensure_columns(df, {"PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB", "GAMES_PLAYED"}, "synthetic player stats")
    return df


def _real_stats() -> pd.DataFrame:
    """Placeholder for real scraping/API logic."""
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
    ensure_columns(df, {"PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB", "GAMES_PLAYED"}, "real player stats")
    return df


def main(use_synthetic: bool = False) -> pd.DataFrame:
    """Fetch player stats and save to PLAYER_STATS_FILE. Returns DataFrame."""
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    try:
        if use_synthetic:
            logger.info("⚙️ Using synthetic player stats for CI/testing...")
            df = _synthetic_stats()
        else:
            logger.info("Fetching real player stats...")
            df = _real_stats()

        df.to_csv(PLAYER_STATS_FILE, index=False)
        logger.info(f"✅ Player stats saved to {PLAYER_STATS_FILE}")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch real stats: {e}")
        logger.info("Falling back to synthetic data...")
        df = _synthetic_stats()
        df.to_csv(PLAYER_STATS_FILE, index=False)
        logger.info(f"Synthetic player stats saved to {PLAYER_STATS_FILE}")
        return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch player stats")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Generate synthetic stats instead of scraping")
    args = parser.parse_args()
    main(use_synthetic=args.use_synthetic)