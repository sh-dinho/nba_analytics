# ============================================================
# File: scripts/generate_dummy_data.py
# Purpose: Generate aligned dummy player_stats.csv and game_results.csv
# ============================================================

import os
import pandas as pd
from core.config import BASE_DATA_DIR, PLAYER_STATS_FILE, GAME_RESULTS_FILE
from core.log_config import setup_logger

logger = setup_logger("generate_dummy_data")

def generate_player_stats():
    """Generate dummy player stats with consistent TEAM_ABBREVIATION values."""
    df = pd.DataFrame({
        "PLAYER_NAME": [
            "LeBron James", "Anthony Davis",
            "Nikola Jokic", "Jamal Murray",
            "Jayson Tatum", "Jaylen Brown",
            "Joel Embiid", "Tyrese Maxey",
            "Stephen Curry", "Klay Thompson"
        ],
        "TEAM_ABBREVIATION": [
            "LAL", "LAL",
            "DEN", "DEN",
            "BOS", "BOS",
            "PHI", "PHI",
            "GSW", "GSW"
        ],
        "AGE": [40, 32, 30, 28, 27, 28, 31, 25, 37, 35],
        "POSITION": ["F", "F", "C", "G", "F", "G", "C", "G", "G", "G"],
        "GAMES_PLAYED": [20, 18, 19, 17, 20, 19, 18, 20, 18, 17],
        "PTS": [25.3, 22.1, 26.4, 21.5, 27.0, 23.2, 29.5, 22.8, 29.1, 18.4],
        "AST": [7.2, 3.5, 9.1, 6.8, 4.1, 3.5, 4.0, 6.2, 6.5, 2.9],
        "REB": [8.1, 10.2, 11.0, 4.2, 7.5, 6.0, 11.3, 3.5, 5.2, 3.8],
    })
    df.to_csv(PLAYER_STATS_FILE, index=False)
    logger.info(f"✅ Dummy player stats saved to {PLAYER_STATS_FILE}")


def generate_game_results():
    """Generate dummy game results with consistent team abbreviations."""
    df = pd.DataFrame([
        {"game_id": 1, "date": "2025-11-25", "home_team": "DEN", "away_team": "LAL", "home_score": 110, "away_score": 102, "home_win": 1},
        {"game_id": 2, "date": "2025-11-25", "home_team": "BOS", "away_team": "PHI", "home_score": 95, "away_score": 100, "home_win": 0},
        {"game_id": 3, "date": "2025-11-26", "home_team": "GSW", "away_team": "LAL", "home_score": 120, "away_score": 115, "home_win": 1},
        {"game_id": 4, "date": "2025-11-26", "home_team": "PHI", "away_team": "DEN", "home_score": 105, "away_score": 99, "home_win": 1},
        {"game_id": 5, "date": "2025-11-27", "home_team": "BOS", "away_team": "GSW", "home_score": 108, "away_score": 112, "home_win": 0},
        {"game_id": 6, "date": "2025-11-28", "home_team": "LAL", "away_team": "PHI", "home_score": 115, "away_score": 118, "home_win": 0},
    ])
    df.to_csv(GAME_RESULTS_FILE, index=False)
    logger.info(f"✅ Dummy game results saved to {GAME_RESULTS_FILE}")


def main():
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    generate_player_stats()
    generate_game_results()
    logger.info("🎉 Dummy data generation complete.")


if __name__ == "__main__":
    main()