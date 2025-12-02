# File: scripts/run_daily_pipeline.py

import os
import sys
import logging
from datetime import datetime

# Import your functions correctly
try:
    from scripts.fetch_player_stats import main as fetch_stats
    from scripts.generate_today_predictions import generate_today_predictions
    from scripts.generate_picks import main as generate_picks
except ImportError:
    # If running as a script, allow relative import
    from .fetch_player_stats import main as fetch_stats
    from .generate_today_predictions import generate_today_predictions
    from .generate_picks import main as generate_picks

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("daily_pipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def run_pipeline(
    season="2024-25",
    resume_stats=True,
    results_dir="results",
    strong_threshold=0.6,
    notify=False
):
    """Run the full daily pipeline: fetch stats → predictions → picks."""

    os.makedirs(results_dir, exist_ok=True)

    # ----------------------------
    # Step 1: Fetch player stats
    # ----------------------------
    logger.info("📊 Step 1: Fetch and update player stats")
    try:
        fetch_stats(season=season, resume=resume_stats)
    except Exception as e:
        logger.error(f"❌ Failed to fetch player stats: {e}")
        return

    # ----------------------------
    # Step 2: Generate today's predictions
    # ----------------------------
    logger.info("📈 Step 2: Generate today's predictions")
    try:
        # Adjusted to match your current function signature
        preds_df = generate_today_predictions(threshold=strong_threshold, notify=notify, outdir=results_dir)
    except Exception as e:
        logger.error(f"❌ Failed to generate predictions: {e}")
        return

    if preds_df is None or preds_df.empty:
        logger.warning("⚠️ No predictions available today.")
        return

    preds_file = os.path.join(results_dir, "predictions.csv")
    preds_df.to_csv(preds_file, index=False)
    logger.info(f"📂 Predictions saved to {preds_file}")

    # ----------------------------
    # Step 3: Generate picks
    # ----------------------------
    logger.info("🎯 Step 3: Generate picks from predictions and odds")
    try:
        generate_picks(
            preds_file=preds_file,
            odds_file=os.path.join("data", "odds.csv"),
            out_file=os.path.join(results_dir, "picks.csv"),
            notify=notify
        )
    except Exception as e:
        logger.error(f"❌ Failed to generate picks: {e}")
        return

    picks_file = os.path.join(results_dir, "picks.csv")
    if not os.path.exists(picks_file):
        logger.error(f"❌ Picks file not found: {picks_file}")
        return

    logger.info(f"✅ Daily pipeline completed. Picks saved to {picks_file}")
    logger.info("📦 Timestamped backups created for predictions and picks")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full NBA daily analytics pipeline")
    parser.add_argument("--season", type=str, default="2024-25", help="NBA season")
    parser.add_argument("--resume_stats", action="store_true", help="Resume fetching stats")
    parser.add_argument("--results_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.6, help="Strong pick probability threshold")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification for top EV pick")
    args = parser.parse_args()

    run_pipeline(
        season=args.season,
        resume_stats=args.resume_stats,
        results_dir=args.results_dir,
        strong_threshold=args.threshold,
        notify=args.notify
    )
