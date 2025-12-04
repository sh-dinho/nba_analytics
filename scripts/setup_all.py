# ============================================================
# File: scripts/setup_all.py
# Purpose: Orchestrate full NBA analytics pipeline with CLI, ledger append, and notifications
# ============================================================

import argparse
import os
import pandas as pd
from pathlib import Path

from core.log_config import init_global_logger
from core.exceptions import PipelineError, DataError
from core.config import (
    NEW_GAMES_FEATURES_FILE,
    TRAINING_FEATURES_FILE,
    PICKS_FILE,
    SUMMARY_FILE as PIPELINE_SUMMARY_FILE,
)

from scripts.fetch_new_games import fetch_new_games
from scripts.build_features_for_training import build_features_for_training
from scripts.build_features_for_new_games import build_features_for_new_games
from scripts.train_model import main as train_model
from scripts.generate_today_predictions import generate_today_predictions
from notifications import send_message, send_photo

logger = init_global_logger()


def _safe_run(step_name, func, *args, **kwargs):
    try:
        logger.info(f"===== Starting: {step_name} =====")
        result = func(*args, **kwargs)
        logger.info(f"✅ Completed: {step_name}")
        return result
    except Exception as e:
        logger.error(f"❌ {step_name} failed: {e}", exc_info=True)
        raise PipelineError(f"{step_name} failed: {e}")


def _append_pipeline_summary(season: str, notes: str, num_picks: int, training_metrics: dict | None):
    run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp": run_time,
        "season": season,
        "target": "pipeline",
        "model_type": "ensemble",  # overall pipeline; training step may refine
        "notes": notes,
        "num_picks": num_picks,
    }

    # Flatten key training metrics if available (e.g., accuracy, logloss, auc)
    if training_metrics:
        for k, v in training_metrics.items():
            # Only add numeric-like values
            try:
                row[f"train_{k}"] = float(v)
            except Exception:
                row[f"train_{k}"] = v

    df = pd.DataFrame([row])

    try:
        path = Path(PIPELINE_SUMMARY_FILE)
        if not path.exists():
            df.to_csv(path, index=False)
            logger.info(f"📑 Created centralized summary at {PIPELINE_SUMMARY_FILE}")
        else:
            df.to_csv(path, mode="a", header=False, index=False)
            logger.info(f"📑 Appended pipeline run to {PIPELINE_SUMMARY_FILE}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to append to centralized summary: {e}")


def main(skip_train=False, skip_fetch=False, season="aggregate", notes="daily pipeline run", notify=False):
    training_metrics = None
    num_picks = 0

    # Step 1: Fetch new games
    if not skip_fetch:
        _safe_run("Fetch New Games", fetch_new_games)
    else:
        logger.info("⏭️ Skipping fetch step per CLI flag")

    # Step 2: Build training features
    if not skip_train:
        _safe_run("Build Features (Training)", build_features_for_training)
    else:
        logger.info("⏭️ Skipping training feature build per CLI flag")

    # Step 3: Train model
    if not skip_train:
        try:
            training_metrics = _safe_run("Train Model", train_model)
            if isinstance(training_metrics, dict) and training_metrics:
                logger.info("=== TRAINING METRICS ===")
                for k, v in training_metrics.items():
                    try:
                        logger.info(f"{k}: {float(v):.4f}")
                    except Exception:
                        logger.info(f"{k}: {v}")
        except PipelineError as e:
            logger.warning(f"⚠️ Training step did not complete: {e}")
    else:
        logger.info("⏭️ Skipping training per CLI flag")

    # Step 4: Build features for new games (fallback to training if none)
    try:
        features_file = _safe_run("Build Features (New Games)", build_features_for_new_games)
    except Exception as e:
        logger.warning(f"⚠️ No new games found or failed to build new-game features: {e}")
        if Path(TRAINING_FEATURES_FILE).exists():
            features_file = TRAINING_FEATURES_FILE
            logger.info("🔁 Falling back to training features for prediction.")
        else:
            raise PipelineError("No features available for prediction (new games or training features missing).")

    # Step 5: Generate today's predictions
    preds = _safe_run(
        "Generate Today's Predictions",
        generate_today_predictions,
        features_file
    )

    # Step 6: EV calculation (graceful handling)
    try:
        if "pred_home_win_prob" in preds.columns and "decimal_odds" in preds.columns:
            preds["expected_value"] = preds["pred_home_win_prob"] * preds["decimal_odds"] - 1
            logger.info("✅ EV calculations completed")
        else:
            logger.info("ℹ️ EV calculation skipped — required columns not present.")
    except Exception as e:
        logger.warning(f"⚠️ EV calculation failed: {e}")

    # Step 7: Picks summary
    try:
        if os.path.exists(PICKS_FILE):
            picks_df = pd.read_csv(PICKS_FILE)
            num_picks = int(len(picks_df))
            if num_picks > 0:
                logger.info(f"🎯 {num_picks} recommended picks saved to {PICKS_FILE}")
            else:
                logger.info("ℹ️ Picks file exists but no positive EV picks today.")
        else:
            logger.info("ℹ️ No picks file generated today.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to read picks file: {e}")

    # Step 8: Append pipeline run to centralized summary
    _append_pipeline_summary(season=season, notes=notes, num_picks=num_picks, training_metrics=training_metrics)

    # Step 9: Optional notifications
    if notify:
        try:
            msg_lines = [
                f"🤖 Pipeline complete ({season})",
                f"Notes: {notes}",
                f"Picks: {num_picks}",
            ]
            if training_metrics:
                # Include a few common metrics if present
                for key in ("accuracy", "auc", "logloss", "f1"):
                    if key in training_metrics:
                        try:
                            val = float(training_metrics[key])
                            msg_lines.append(f"{key.upper()}: {val:.4f}")
                        except Exception:
                            msg_lines.append(f"{key.upper()}: {training_metrics[key]}")

            send_message("\n".join(msg_lines))
            logger.info("📲 Pipeline summary pushed to Telegram")
        except Exception as e:
            logger.warning(f"⚠️ Failed to send Telegram notification: {e}")

    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate full NBA analytics pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetch step")
    parser.add_argument("--season", type=str, default="aggregate", help="Season tag for pipeline run (e.g., 2025-26)")
    parser.add_argument("--notes", type=str, default="daily pipeline run", help="Optional notes to annotate run")
    parser.add_argument("--notify", action="store_true", help="Send summary to Telegram")
    args = parser.parse_args()

    main(
        skip_train=args.skip_train,
        skip_fetch=args.skip_fetch,
        season=args.season,
        notes=args.notes,
        notify=args.notify,
    )
