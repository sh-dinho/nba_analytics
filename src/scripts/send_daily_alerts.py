from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Daily Alert Dispatcher (Canonical)
# File: src/scripts/send_daily_alerts.py
# Author: Sadiq
#
# Description:
#     Unified daily alert runner for the canonical pipeline:
#       • Generates data quality dashboard
#       • Sends data quality summary
#       • Sends model monitoring summary
#       • Sends betting recommendations (if available)
#       • Sends bankroll chart (optional)
#
#     All alerts routed through AlertManager.
# ============================================================

import json
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from src.alerts.alert_manager import AlertManager
from src.alerts.recommendations_alert import send_recommendations_alert
from src.scripts.generate_data_quality_dashboard import generate_dashboard
from src.config.paths import LOGS_DIR, BET_LOG_PATH, RECOMMENDATIONS_DIR


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_dashboard() -> dict:
    """Load the most recent canonical dashboard JSON."""
    path = LOGS_DIR / "dashboard_latest.json"
    if not path.exists():
        raise RuntimeError("Dashboard JSON not found. Run generator first.")
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to read dashboard JSON: {e}")


def load_recommendations(pred_date: str):
    """
    Load canonical recommendations.
    Expected file: RECOMMENDATIONS_DIR / recs_YYYY-MM-DD.parquet
    """
    rec_path = RECOMMENDATIONS_DIR / f"recs_{pred_date}.parquet"
    if rec_path.exists():
        try:
            return pd.read_parquet(rec_path)
        except Exception as e:
            logger.error(f"Failed to read recommendations file: {e}")
            return None
    return None


def load_bankroll_history():
    """Load canonical bankroll history."""
    if BET_LOG_PATH.exists():
        try:
            return pd.read_csv(BET_LOG_PATH, parse_dates=["date"])
        except Exception as e:
            logger.error(f"Failed to read bankroll history: {e}")
            return None
    return None


# ------------------------------------------------------------
# Main dispatcher
# ------------------------------------------------------------

def run_daily_alerts() -> dict:
    logger.info("=== Running Daily Alert Dispatcher (Canonical) ===")

    alerts = AlertManager()
    today = date.today().strftime("%Y-%m-%d")

    # --------------------------------------------------------
    # 1. Generate dashboard JSON
    # --------------------------------------------------------
    try:
        generate_dashboard()
        dashboard = load_dashboard()
    except Exception as e:
        msg = f"Failed to generate or load dashboard: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # 2. Send data quality summary
    # --------------------------------------------------------
    try:
        alerts.alert_data_quality(dashboard)
    except Exception as e:
        logger.error(f"Failed to send data quality alert: {e}")

    # --------------------------------------------------------
    # 3. Send model monitoring summary
    # --------------------------------------------------------
    try:
        if "model_monitoring" in dashboard:
            alerts.alert_model_monitor(dashboard["model_monitoring"])
        else:
            logger.warning("Dashboard missing model_monitoring section.")
    except Exception as e:
        logger.error(f"Failed to send model monitoring alert: {e}")

    # --------------------------------------------------------
    # 4. Send betting recommendations (if available)
    # --------------------------------------------------------
    try:
        recs = load_recommendations(today)
        if recs is not None and not recs.empty:
            send_recommendations_alert(recs, today)
        else:
            logger.info("No betting recommendations found for today.")
    except Exception as e:
        logger.error(f"Failed to send recommendations alert: {e}")

    # --------------------------------------------------------
    # 5. Send bankroll chart (optional)
    # --------------------------------------------------------
    try:
        history = load_bankroll_history()
        if history is not None and len(history) > 5:
            alerts.telegram.send_bankroll_chart(history)
            logger.info("Bankroll chart sent.")
        else:
            logger.info("No bankroll history available.")
    except Exception as e:
        logger.error(f"Failed to send bankroll chart: {e}")

    logger.success("=== Daily Alerts Completed ===")

    return {
        "ok": True,
        "date": today,
        "dashboard_loaded": True,
        "recommendations_sent": recs is not None and not recs.empty,
        "bankroll_chart_sent": history is not None and len(history) > 5,
    }


def main():
    run_daily_alerts()


if __name__ == "__main__":
    main()