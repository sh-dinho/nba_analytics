from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Daily Model Monitoring Job
# File: src/scripts/monitor_daily.py
# Author: Sadiq
#
# Description:
#     Daily cron‑ready monitoring job that:
#       • Loads latest predictions
#       • Runs ModelMonitor(predictions=df)
#       • Writes JSON summary to LOGS_DIR
#       • Prints human‑readable summary
#
#     Compatible with the modern canonical pipeline.
# ============================================================

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.monitoring.model_monitor import ModelMonitor
from src.config.paths import LOGS_DIR, PREDICTIONS_DIR


def _load_latest_predictions() -> pd.DataFrame:
    """Load the most recent predictions file from PREDICTIONS_DIR."""
    if not PREDICTIONS_DIR.exists():
        raise FileNotFoundError(f"PREDICTIONS_DIR does not exist: {PREDICTIONS_DIR}")

    files = sorted(PREDICTIONS_DIR.glob("predictions_*.parquet"))
    if not files:
        raise FileNotFoundError("No prediction files found in PREDICTIONS_DIR")

    latest = files[-1]
    logger.info(f"Loading latest predictions: {latest}")
    return pd.read_parquet(latest)


def run_daily_monitor() -> dict:
    """Run the daily model monitoring job and return a structured report."""
    logger.info("=== Running Daily Model Monitoring ===")

    # --------------------------------------------------------
    # Load latest predictions
    # --------------------------------------------------------
    try:
        preds = _load_latest_predictions()
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        return {
            "ok": False,
            "error": f"Failed to load predictions: {e}",
            "timestamp_utc": datetime.utcnow().isoformat(),
        }

    # --------------------------------------------------------
    # Run ModelMonitor
    # --------------------------------------------------------
    try:
        monitor = ModelMonitor(predictions=preds)
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
    out_path = LOGS_DIR / f"monitor_report_{timestamp}.json"

    try:
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.success(f"Daily monitor report saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to write monitor report: {e}")

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------
    print("\n=== DAILY MONITOR SUMMARY ===")
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
    run_daily_monitor()


if __name__ == "__main__":
    main()