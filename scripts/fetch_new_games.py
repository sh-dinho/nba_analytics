# ============================================================
# File: scripts/fetch_new_games.py
# Purpose: Fetch today's games or generate synthetic games for CI/testing
# ============================================================

import os
import pandas as pd
from core.config import NEW_GAMES_FILE, BASE_DATA_DIR
from core.log_config import setup_logger
from core.utils import ensure_columns

logger = setup_logger("fetch_new_games")


def fetch_new_games(use_synthetic: bool = False) -> str:
    """
    Fetch today's games. If use_synthetic=True, generate synthetic games for CI/testing.
    Saves to NEW_GAMES_FILE with raw stats (feature engineering added later).
    Returns the path to the saved file.
    """
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    if use_synthetic:
        logger.info("⚙️ Using synthetic new_games.csv for CI/testing...")
        df = pd.DataFrame([
            {"PLAYER_NAME": "SYN_Player1", "TEAM_ABBREVIATION": "SYN_A", "TEAM_HOME": "SYN_A", "TEAM_AWAY": "SYN_B",
             "PTS": 15, "AST": 5, "REB": 4, "GAMES_PLAYED": 10, "decimal_odds": 1.8},
            {"PLAYER_NAME": "SYN_Player2", "TEAM_ABBREVIATION": "SYN_A", "TEAM_HOME": "SYN_A", "TEAM_AWAY": "SYN_B",
             "PTS": 18, "AST": 6, "REB": 5, "GAMES_PLAYED": 12, "decimal_odds": 1.8},
            {"PLAYER_NAME": "SYN_Player3", "TEAM_ABBREVIATION": "SYN_B", "TEAM_HOME": "SYN_A", "TEAM_AWAY": "SYN_B",
             "PTS": 20, "AST": 7, "REB": 6, "GAMES_PLAYED": 12, "decimal_odds": 2.1},
            {"PLAYER_NAME": "SYN_Player4", "TEAM_ABBREVIATION": "SYN_B", "TEAM_HOME": "SYN_A", "TEAM_AWAY": "SYN_B",
             "PTS": 22, "AST": 8, "REB": 7, "GAMES_PLAYED": 14, "decimal_odds": 2.1},
        ])
        df.to_csv(NEW_GAMES_FILE, index=False)
        logger.info(f"✅ Synthetic new_games.csv saved to {NEW_GAMES_FILE}")
        return NEW_GAMES_FILE

    # --- Real scraping logic placeholder ---
    try:
        logger.info("📡 Fetching live data (implement your fetch logic here)...")
        df = pd.DataFrame([
            {"PLAYER_NAME": "LeBron James", "TEAM_ABBREVIATION": "LAL", "TEAM_HOME": "LAL", "TEAM_AWAY": "BOS",
             "PTS": 25, "AST": 7, "REB": 8, "GAMES_PLAYED": 20, "decimal_odds": 1.9},
            {"PLAYER_NAME": "Anthony Davis", "TEAM_ABBREVIATION": "LAL", "TEAM_HOME": "LAL", "TEAM_AWAY": "BOS",
             "PTS": 23, "AST": 4, "REB": 10, "GAMES_PLAYED": 18, "decimal_odds": 1.9},
            {"PLAYER_NAME": "Jayson Tatum", "TEAM_ABBREVIATION": "BOS", "TEAM_HOME": "LAL", "TEAM_AWAY": "BOS",
             "PTS": 27, "AST": 5, "REB": 7, "GAMES_PLAYED": 19, "decimal_odds": 2.2},
            {"PLAYER_NAME": "Jaylen Brown", "TEAM_ABBREVIATION": "BOS", "TEAM_HOME": "LAL", "TEAM_AWAY": "BOS",
             "PTS": 22, "AST": 3, "REB": 6, "GAMES_PLAYED": 18, "decimal_odds": 2.2},
            {"PLAYER_NAME": "Stephen Curry", "TEAM_ABBREVIATION": "GSW", "TEAM_HOME": "GSW", "TEAM_AWAY": "MIA",
             "PTS": 29, "AST": 6, "REB": 5, "GAMES_PLAYED": 18, "decimal_odds": 2.0},
            {"PLAYER_NAME": "Klay Thompson", "TEAM_ABBREVIATION": "GSW", "TEAM_HOME": "GSW", "TEAM_AWAY": "MIA",
             "PTS": 21, "AST": 3, "REB": 4, "GAMES_PLAYED": 17, "decimal_odds": 2.0},
            {"PLAYER_NAME": "Jimmy Butler", "TEAM_ABBREVIATION": "MIA", "TEAM_HOME": "GSW", "TEAM_AWAY": "MIA",
             "PTS": 24, "AST": 6, "REB": 7, "GAMES_PLAYED": 18, "decimal_odds": 1.95},
            {"PLAYER_NAME": "Bam Adebayo", "TEAM_ABBREVIATION": "MIA", "TEAM_HOME": "GSW", "TEAM_AWAY": "MIA",
             "PTS": 19, "AST": 4, "REB": 10, "GAMES_PLAYED": 18, "decimal_odds": 1.95},
        ])

        # Validate required columns
        ensure_columns(df, {"PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_HOME", "TEAM_AWAY", "PTS", "AST", "REB", "GAMES_PLAYED"}, "new games")

        df.to_csv(NEW_GAMES_FILE, index=False)
        logger.info(f"✅ Real new_games.csv saved to {NEW_GAMES_FILE}")
        return NEW_GAMES_FILE
    except Exception as e:
        logger.error(f"❌ Failed to fetch real games: {e}")
        logger.info("Falling back to synthetic games...")
        return fetch_new_games(use_synthetic=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch today's games")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Generate synthetic games instead of scraping")
    args = parser.parse_args()
    fetch_new_games(use_synthetic=args.use_synthetic)