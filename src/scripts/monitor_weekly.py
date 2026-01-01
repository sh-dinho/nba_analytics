from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Weekly Model Monitoring Job
# File: src/scripts/monitor_weekly.py
# Author: Sadiq
#
# Description:
#     Weekly cron‑ready monitoring job that:
#       • Loads all predictions from the last 7 days
#       • Runs ModelMonitor(predictions=df)
#       • Writes JSON summary to LOGS_DIR
#       • Prints human‑readable summary
# ============================================================

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from src.monitoring.model_monitor import ModelMonitor
from src.config.paths import LOGS_DIR, PREDICTIONS_DIR


# ------------------------------------------------------------
# Load predictions (supports legacy + modern filenames)
# ------------------------------------------------------------

def load_last_7_days_predictions() -> pd.DataFrame:
    cutoff = datetime.utcnow() - timedelta(days=7)
    dfs = []

    if not PREDICTIONS_DIR.exists():
        logger.error(f"PREDICTIONS_DIR does not exist: {PREDICTIONS_DIR}")
        return pd.DataFrame()

    for file in sorted(PREDICTIONS_DIR.glob("predictions_*.parquet")):
        try:
            name = file.name.replace(".parquet", "")

            # Modern format: predictions_YYYY-MM-DD
            if "_" in name and "-" in name:
                date_str = name.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Legacy format: predictions_YYYYMMDD_vX
            else:
                date_str = name.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y%m%d")

            if file_date >= cutoff:
                df = pd.read_parquet(file)
                dfs.append(df)

        except Exception as e:
            logger.warning(f"Skipping file {file.name}: {e}")

    if not dfs:
        logger.warning("No prediction files found for the last 7 days.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} predictions from the last 7 days.")
    return combined


# ------------------------------------------------------------
# Main weekly monitor
# ------------------------------------------------------------

def run_weekly_monitor() -> dict:
    logger.info("=== Running Weekly Model Monitoring ===")

    predictions_df = load_last_7_days_predictions()

    try:
        monitor = ModelMonitor(predictions=predictions_df)
        report = monitor.run()
    except Exception as e:
        logger.error(f"ModelMonitor failed: {e}")
        return {
            "ok": False,
            "error": f"ModelMonitor failed: {e}",
            "timestamp_utc": datetime.utcnow().isoformat(),
        }

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = LOGS_DIR / f"monitor_weekly_{timestamp}.json"

    try:
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.success(f"Weekly monitor report saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to write weekly monitor report: {e}")

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------
    print("\n=== WEEKLY MONITOR SUMMARY ===")
    print(f"OK: {getattr(report, 'ok', False)}")

    print("\nIssues:")
    issues = getattr(report, "issues", [])
    if not issues:
        print(" - None")
    else:
        for issue in issues:
            level = getattr(issue, "level", "unknown").upper()
            msg = getattr(issue, "message", "")
            details = getattr(issue, "details", "")
            print(f" - [{level}] {msg} | {details}")

    print("\nMetrics:")
    metrics = getattr(report, "metrics", {})
    if not metrics:
        print(" - None")
    else:
        for k, v in metrics.items():
            print(f" - {k}: {v}")

    print("\n=== DONE ===")

    return {
        "ok": getattr(report, "ok", False),
        "issues": [i.to_dict() for i in issues] if issues else [],
        "metrics": metrics,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "output_path": str(out_path),
    }


def main():
    run_weekly_monitor()


if __name__ == "__main__":
    main()