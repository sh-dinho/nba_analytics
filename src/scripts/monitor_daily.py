from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Daily Model Monitoring Job
# File: src/scripts/monitor_daily.py
# Author: Sadiq
# ============================================================

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.monitoring.model_monitor import ModelMonitor
from src.config.paths import LOGS_DIR, PREDICTIONS_DIR
from src.config.env import MODEL_VERSION, MODEL_ENVIRONMENT


# ------------------------------------------------------------
# Load Latest Predictions
# ------------------------------------------------------------
def _load_latest_predictions() -> tuple[pd.DataFrame, Path]:
    """Load the most recent predictions file from PREDICTIONS_DIR."""
    if not PREDICTIONS_DIR.exists():
        raise FileNotFoundError(f"PREDICTIONS_DIR does not exist: {PREDICTIONS_DIR}")

    files = sorted(PREDICTIONS_DIR.glob("predictions_*.parquet"))
    if not files:
        raise FileNotFoundError("No prediction files found in PREDICTIONS_DIR")

    latest = files[-1]
    logger.info(f"Loading latest predictions: {latest}")

    df = pd.read_parquet(latest)

    if df.empty:
        raise ValueError(f"Predictions file is empty: {latest}")

    required_cols = {"game_id", "team", "opponent", "win_probability"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    return df, latest


# ------------------------------------------------------------
# Daily Monitor
# ------------------------------------------------------------
def run_daily_monitor() -> dict:
    """Run the daily model monitoring job and return a structured report."""
    logger.info("=== Running Daily Model Monitoring ===")

    # --------------------------------------------------------
    # Load latest predictions
    # --------------------------------------------------------
    try:
        preds, pred_path = _load_latest_predictions()
    except Exception as e:
        logger.exception("Failed to load predictions")
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
        report_dict = report.to_dict()
    except Exception as e:
        logger.exception("ModelMonitor failed")
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

    enriched_report = {
        "ok": report_dict.get("ok", False),
        "issues": report_dict.get("issues", []),
        "metrics": report_dict.get("metrics", {}),
        "model_version": MODEL_VERSION,
        "model_environment": MODEL_ENVIRONMENT,
        "prediction_file": str(pred_path),
        "timestamp_utc": datetime.utcnow().isoformat(),
    }

    try:
        out_path.write_text(json.dumps(enriched_report, indent=2))
        logger.success(f"Daily monitor report saved to {out_path}")
    except Exception as e:
        logger.exception("Failed to write monitor report")

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------
    print("\n=== DAILY MONITOR SUMMARY ===")
    print(f"OK: {enriched_report['ok']}")

    print("\nIssues:")
    issues = enriched_report["issues"]
    if not issues:
        print(" - None")
    else:
        for issue in issues:
            level = issue.get("level", "unknown").upper()
            msg = issue.get("message", "")
            details = issue.get("details", "")
            print(f" - [{level}] {msg} | {details}")

    print("\nMetrics:")
    metrics = enriched_report["metrics"]
    if not metrics:
        print(" - None")
    else:
        for k, v in metrics.items():
            print(f" - {k}: {v}")

    print("\n=== DONE ===")

    return enriched_report


def main():
    run_daily_monitor()


if __name__ == "__main__":
    main()
