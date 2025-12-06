# ============================================================
# Build features for new games and save to CSV
# ============================================================

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nba_core.paths import NEW_GAMES_FEATURES_FILE, ensure_dirs
from nba_core.log_config import init_global_logger

logger = init_global_logger("build_features")

def build_new_game_features():
    ensure_dirs(strict=False)

    # Example: load raw schedule or odds feed
    # Replace with your ingestion logic
    raw_data = pd.DataFrame([
        {"GAME_ID": 20251206, "HOME_TEAM": "Boston Celtics", "AWAY_TEAM": "Los Angeles Lakers", "TEAM_ID": "BOS", "decimal_odds": 1.85},
        {"GAME_ID": 20251207, "HOME_TEAM": "Miami Heat", "AWAY_TEAM": "Chicago Bulls", "TEAM_ID": "MIA", "decimal_odds": 2.10},
    ])

    # Construct GAME_NAME column
    raw_data["GAME_NAME"] = raw_data["HOME_TEAM"] + " vs " + raw_data["AWAY_TEAM"]

    # Save features file
    raw_data.to_csv(NEW_GAMES_FEATURES_FILE, index=False)
    logger.info(f"✅ New game features built → {NEW_GAMES_FEATURES_FILE}")

if __name__ == "__main__":
    build_new_game_features()
