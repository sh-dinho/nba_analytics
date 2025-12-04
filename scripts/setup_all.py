# ============================================================
# File: scripts/setup_all.py
# Purpose: Full NBA pipeline setup: fetch games, build features, generate predictions
# ============================================================

import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------
# Core imports
# -----------------------------
from core.paths import ensure_dirs, DATA_DIR, FEATURES_FILE
from core.config import PREDICTIONS_FILE, PREDICTION_THRESHOLD
from core.log_config import init_global_logger
from notifications import send_telegram_message

# -----------------------------
# Scripts imports
# -----------------------------
from scripts.fetch_new_games import fetch_new_games
from scripts.generate_today_predictions import generate_today_predictions

# Initialize logger
logger = init_global_logger()

def main():
    logger.info("🚀 Starting NBA analytics pipeline...")

    # Ensure necessary directories
    ensure_dirs(strict=False)
    logger.info("✅ All pipeline directories exist.")

    # Step 1: Fetch today's NBA games
    try:
        new_games_file = fetch_new_games(debug=False, allow_placeholder=True)
        logger.info(f"📥 Fetched new games → {new_games_file}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch new games: {e}")
        send_telegram_message(f"❌ Pipeline failed at fetch_new_games: {e}")
        return

    # Step 2: Generate predictions
    try:
        df_predictions = generate_today_predictions(
            features_file=str(FEATURES_FILE),
            threshold=PREDICTION_THRESHOLD
        )
        logger.info(f"📊 Predictions generated → {PREDICTIONS_FILE} ({len(df_predictions)} rows)")
    except Exception as e:
        logger.error(f"❌ Failed to generate predictions: {e}")
        send_telegram_message(f"❌ Pipeline failed at generate_today_predictions: {e}")
        return

    logger.info
