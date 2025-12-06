# ============================================================
# File: generate_features.py
# Purpose: Safely generate team & player feature CSVs for modeling
# ============================================================

from nba_core.paths import (
    TRAINING_FEATURES_FILE,
    PLAYER_FEATURES_FILE,
    NEW_GAMES_FEATURES_FILE,
    ensure_dirs,
)
from nba_core.log_config import init_global_logger
from nba_core.exceptions import FileError, DataError
from features.feature_builder import (
    build_training_features,
    build_upcoming_features,
    build_player_features,
)

logger = init_global_logger()
ensure_dirs(strict=False)

# ---------------- Entrypoint ----------------
def main():
    logger.info("🏀 Generating NBA Features...")

    # Generate training (team) features
    try:
        team_df = build_training_features()
    except (FileError, DataError) as e:
        logger.error(f"❌ Error generating training features: {e}")
        team_df = None

    # Generate upcoming game features
    try:
        upcoming_df = build_upcoming_features()
    except (FileError, DataError) as e:
        logger.error(f"❌ Error generating upcoming game features: {e}")
        upcoming_df = None

    # Generate player features
    try:
        player_df = build_player_features()
    except (FileError, DataError) as e:
        logger.error(f"❌ Error generating player features: {e}")
        player_df = None

    logger.info("✅ Feature generation complete.")
    logger.info(
        f"Summary → Training: {TRAINING_FEATURES_FILE}, "
        f"Upcoming: {NEW_GAMES_FEATURES_FILE}, "
        f"Player: {PLAYER_FEATURES_FILE}"
    )

    return {"training": team_df, "upcoming": upcoming_df, "player": player_df}


if __name__ == "__main__":
    main()
