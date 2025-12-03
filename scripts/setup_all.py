# ============================================================
# File: scripts/setup_all.py
# Purpose: Orchestrate full NBA analytics pipeline
# ============================================================

import argparse
from core.log_config import setup_logger
from core.exceptions import PipelineError

from scripts.fetch_new_games import fetch_new_games
from scripts.build_features_for_training import build_features_for_training
from scripts.build_features_for_new_games import build_features_for_new_games
from scripts.train_model import main as train_model
from scripts.generate_today_predictions import generate_today_predictions
from core.config import NEW_GAMES_FEATURES_FILE, TRAINING_FEATURES_FILE

logger = setup_logger("setup_all")


def _safe_run(step_name, func, *args, **kwargs):
    try:
        logger.info(f"===== Starting: {step_name} =====")
        result = func(*args, **kwargs)
        logger.info(f"✅ Completed: {step_name}")
        return result
    except Exception as e:
        logger.error(f"❌ {step_name} failed: {e}", exc_info=True)
        raise PipelineError(f"{step_name} failed: {e}")


def main(skip_train=False, skip_fetch=False):
    # Step 1: Fetch new games
    if not skip_fetch:
        _safe_run("Fetch New Games", fetch_new_games)

    # Step 2: Build training features
    if not skip_train:
        _safe_run("Build Features (Training)", build_features_for_training)

    # Step 3: Train model
    if not skip_train:
        metrics = _safe_run("Train Model", train_model)
        logger.info("=== TRAINING METRICS ===")
        for k, v in metrics.items():
            logger.info(f"{k.capitalize()}: {v:.3f}")

    # Step 4: Build features for new games (fallback to training if none)
    try:
        features_file = _safe_run("Build Features (New Games)", build_features_for_new_games)
    except FileNotFoundError:
        logger.warning("⚠️ No new games found today — falling back to historical training features")
        features_file = TRAINING_FEATURES_FILE

    # Step 5: Generate today's predictions
    preds = _safe_run(
        "Generate Today's Predictions",
        generate_today_predictions,
        features_file
    )

    # Step 6: EV calculation (graceful handling)
    if "decimal_odds" in preds.columns:
        preds["expected_value"] = preds["pred_home_win_prob"] * preds["decimal_odds"] - 1
        logger.info("✅ EV calculations completed")
    else:
        logger.warning("⚠️ 'decimal_odds' column missing — skipping EV calculations")

    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetch step")
    args = parser.parse_args()

    main(skip_train=args.skip_train, skip_fetch=args.skip_fetch)