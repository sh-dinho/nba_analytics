# ============================================================
# File: generate_features.py
# Purpose: Safely generate team & player feature CSVs for modeling
# ============================================================

import pandas as pd
from pathlib import Path
from core.config import HISTORICAL_GAMES_FILE, PLAYER_GAMES_FILE, TRAINING_FEATURES_FILE, PLAYER_FEATURES_FILE
from core.log_config import init_global_logger
from features.feature_builder import build_team_features, build_player_features

logger = init_global_logger()

# ---------------- Entrypoint ----------------
def main():
    logger.info("🏀 Generating NBA Features...")

    # Generate team features
    try:
        generate_team_features()
    except Exception as e:
        logger.error(f"Error generating team features: {e}")

    # Generate player features
    try:
        generate_player_features()
    except Exception as e:
        logger.error(f"Error generating player features: {e}")
    
    logger.info("✅ Feature generation complete.")

# ---------------- Team Features ----------------
def generate_team_features():
    """Generate and save team features."""
    df_features = build_team_features(out_file=TRAINING_FEATURES_FILE)
    logger.info(f"📂 Team features saved → {TRAINING_FEATURES_FILE}")
    return df_features

# ---------------- Player Features ----------------
def generate_player_features():
    """Generate and save player features."""
    df_features = build_player_features(out_file=PLAYER_FEATURES_FILE)
    logger.info(f"📂 Player features saved → {PLAYER_FEATURES_FILE}")
    return df_features

if __name__ == "__main__":
    main()
