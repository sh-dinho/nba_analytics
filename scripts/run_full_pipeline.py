# File: scripts/run_full_pipeline.py

import logging
import sys
from datetime import datetime
import os

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


# ===============================================================
# Helpers
# ===============================================================

def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _safe_step(name: str, func, fail_fast: bool, *args, **kwargs):
    """
    Standard wrapper for pipeline steps.

    Returns:
        (result, success_flag)
    """
    logger.info(f"\n===== 🚀 Starting Step: {name} =====")
    try:
        out = func(*args, **kwargs)
        logger.info(f"✅ Completed: {name}")
        return out, True
    except Exception as e:
        logger.error(f"❌ {name} failed: {e}")
        if fail_fast:
            raise
        return None, False


# ===============================================================
# Main Pipeline
# ===============================================================

def run_pipeline(season="2024-25", fail_fast=False):
    """
    Full data pipeline:
      1. Fetch player stats
      2. Compare snapshots
      3. Build weekly summaries
      4. Build training dataset
    """
    start_ts = _timestamp()
    logger.info(f"\n🚀 Starting Full NBA Analytics Pipeline at {start_ts}")
    logger.info(f"Processing season: {season}")

    # Ensure required directories exist
    for d in ["data", "results", "models", "snapshots"]:
        os.makedirs(d, exist_ok=True)

    success = True  # track global success

    # ------------------
    # 1️⃣ Player Stats
    # ------------------
    _, ok = _safe_step(
        "Fetch Player Stats (Parallel)",
        fps.fetch_and_save_player_stats_parallel,
        fail_fast,
        season=season,
        resume=True,
        max_workers=8
    )
    success &= ok

    # ------------------
    # 2️⃣ Snapshot Trends
    # ------------------
    _, ok = _safe_step(
        "Compare Snapshots",
        compare_snapshots.compare_snapshots,
        fail_fast,
        notify=True
    )
    success &= ok

    # ------------------
    # 3️⃣ Weekly Summary
    # ------------------
    _, ok = _safe_step(
        "Build Weekly Summary",
        build_weekly_summary.build_weekly_summary,
        fail_fast,
        notify=True
    )
    success &= ok

    # ------------------
    # 4️⃣ Build Training Data
    # ------------------
    _, ok = _safe_step(
        "Build Training Dataset",
        build_training_data.build_training_data,
        fail_fast,
        scale=True
    )
    success &= ok

    # ------------------
    # Wrap Up
    # ------------------
    end_ts = _timestamp()
    logger.info(f"\n🎉 Full Pipeline Completed at {end_ts}")
    logger.info(f"⏱ Total duration: {datetime.now() - datetime.strptime(start_ts, '%Y-%m-%d_%H-%M-%S')}")

    if success:
        logger.info("✅ All steps completed successfully\n")
    else:
        logger.warning("⚠️ Pipeline finished with one or more step failures\n")

    return success


# ===============================================================
# CLI Entrypoint
# ===============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full NBA analytics data pipeline")
    parser.add_argument("--season", type=str, default="2024-25", help="Season to process, e.g., 2024-25")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop pipeline immediately if any step fails"
    )

    args = parser.parse_args()

    try:
        ok = run_pipeline(season=args.season, fail_fast=args.fail_fast)
        if not ok:
            sys.exit(2)
    except Exception as e:
        logger.error(f"❌ Full pipeline crashed: {e}")
        sys.exit(1)
