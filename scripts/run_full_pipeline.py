# File: scripts/run_full_pipeline.py
import logging
import sys
from datetime import datetime
from scripts import fetch_player_stats_parallel as fps
from scripts import compare_snapshots
from scripts import build_weekly_summary
from scripts import build_training_data

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("full_pipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

def run_pipeline(season="2024-25"):
    logger.info("🚀 Starting full NBA analytics pipeline...")

    # Step 1: Fetch player stats (parallel + retry support)
    logger.info("1️⃣ Fetching player stats...")
    fps.fetch_and_save_player_stats_parallel(season=season, resume=True, max_workers=8)

    # Step 2: Compare snapshots and generate trends
    logger.info("2️⃣ Comparing snapshots to build trends...")
    compare_snapshots.compare_snapshots(notify=True)

    # Step 3: Build weekly team summary
    logger.info("3️⃣ Building weekly summary...")
    build_weekly_summary.build_weekly_summary(notify=True)

    # Step 4: Build training dataset
    logger.info("4️⃣ Building training dataset for ML...")
    build_training_data.build_training_data(scale=True)

    logger.info(f"✅ Pipeline finished at {datetime.now().isoformat()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full NBA analytics pipeline")
    parser.add_argument("--season", type=str, default="2024-25", help="Season to process (e.g., 2024-25)")
    args = parser.parse_args()

    try:
        run_pipeline(season=args.season)
    except Exception as e:
        logger.error(f"❌ Full pipeline failed: {e}")
        sys.exit(1)
