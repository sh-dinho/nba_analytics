# ============================================================
# File: scripts/generate_synthetic_historical_games.py
# Purpose: Generate a synthetic historical_games.csv for testing
# ============================================================

import os
import pandas as pd
from core.config import HISTORICAL_GAMES_FILE, BASE_DATA_DIR
from core.log_config import setup_logger

logger = setup_logger("generate_synthetic_historical_games")


def generate_synthetic_historical_games() -> str:
    """
    Generate a synthetic historical_games.csv with a few rows.
    Includes PLAYER_NAME, TEAM_ABBREVIATION, TEAM_HOME, TEAM_AWAY,
    PTS, AST, REB, GAMES_PLAYED, HOME_WIN.
    """
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    data = [
        # Lakers vs Celtics (LAL wins)
        {"PLAYER_NAME": "LeBron James", "TEAM_ABBREVIATION": "LAL", "TEAM_HOME": "LAL", "TEAM_AWAY": "BOS",
         "PTS": 28, "AST": 8, "REB": 9, "GAMES_PLAYED": 20, "HOME_WIN": 1},
        {"PLAYER_NAME": "Anthony Davis", "TEAM_ABBREVIATION": "LAL", "TEAM_HOME": "LAL", "TEAM_AWAY": "BOS",
         "PTS": 24, "AST": 3, "REB": 11, "GAMES_PLAYED": 18, "HOME_WIN": 1},
        {"PLAYER_NAME": "Jayson Tatum", "TEAM_ABBREVIATION": "BOS", "TEAM_HOME": "LAL", "TEAM_AWAY": "BOS",
         "PTS": 26, "AST": 5, "REB": 7, "GAMES_PLAYED": 19, "HOME_WIN": 1},
        {"PLAYER_NAME": "Jaylen Brown", "TEAM_ABBREVIATION": "BOS", "TEAM_HOME": "LAL", "TEAM_AWAY": "BOS",
         "PTS": 22, "AST": 4, "REB": 6, "GAMES_PLAYED": 18, "HOME_WIN": 1},

        # Warriors vs Heat (MIA wins)
        {"PLAYER_NAME": "Stephen Curry", "TEAM_ABBREVIATION": "GSW", "TEAM_HOME": "GSW", "TEAM_AWAY": "MIA",
         "PTS": 30, "AST": 7, "REB": 5, "GAMES_PLAYED": 21, "HOME_WIN": 0},
        {"PLAYER_NAME": "Klay Thompson", "TEAM_ABBREVIATION": "GSW", "TEAM_HOME": "GSW", "TEAM_AWAY": "MIA",
         "PTS": 20, "AST": 2, "REB": 4, "GAMES_PLAYED": 20, "HOME_WIN": 0},
        {"PLAYER_NAME": "Jimmy Butler", "TEAM_ABBREVIATION": "MIA", "TEAM_HOME": "GSW", "TEAM_AWAY": "MIA",
         "PTS": 25, "AST": 6, "REB": 7, "GAMES_PLAYED": 19, "HOME_WIN": 0},
        {"PLAYER_NAME": "Bam Adebayo", "TEAM_ABBREVIATION": "MIA", "TEAM_HOME": "GSW", "TEAM_AWAY": "MIA",
         "PTS": 18, "AST": 4, "REB": 10, "GAMES_PLAYED": 19, "HOME_WIN": 0},
    ]

    df = pd.DataFrame(data)
    df.to_csv(HISTORICAL_GAMES_FILE, index=False)
    logger.info(f"✅ Synthetic historical_games.csv saved to {HISTORICAL_GAMES_FILE} ({len(df)} rows)")
    return HISTORICAL_GAMES_FILE


if __name__ == "__main__":
    generate_synthetic_historical_games()