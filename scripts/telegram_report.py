# ============================================================
# File: scripts/telegram_report.py
# Purpose: Send bankroll summary + chart + trend analysis to Telegram
# ============================================================

import os
import requests
import pandas as pd
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError

logger = setup_logger("telegram_report")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_message(text: str):
    """Send a text message to Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set. Skipping report.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("📲 Telegram text report sent successfully")
    except Exception as e:
        logger.error(f"❌ Failed to send Telegram text report: {e}")
        raise PipelineError(f"Telegram message failed: {e}")


def send_photo(photo_path: str, caption: str = None):
    """Send a photo to Telegram with optional caption."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set. Skipping photo upload.")
        return
    if not os.path.exists(photo_path):
        logger.warning(f"⚠️ Chart not found at {photo_path}. Skipping photo upload.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        files = {"photo": photo}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption or ""}
        try:
            resp = requests.post(url, data=data, files=files, timeout=10)
            resp.raise_for_status()
            logger.info("📸 Telegram chart sent successfully")
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram chart: {e}")
            raise PipelineError(f"Telegram photo failed: {e}")


def main():
    """Load combined summary, format report, and send to Telegram."""
    summary_path = "results/combined_summary.csv"
    chart_path = "results/bankroll_comparison.png"

    if not os.path.exists(summary_path):
        logger.warning("⚠️ No combined_summary.csv found. Skipping report.")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        raise DataError(f"Failed to read combined_summary.csv: {e}")

    if df.empty:
        logger.warning("⚠️ Combined summary is empty. Skipping report.")
        return

    # Format summary message
    message = "*🏀 NBA Daily Combined Report*\n\n"
    for _, row in df.iterrows():
        try:
            message += (
                f"📌 Model: {row['Model']}\n"
                f"🏦 Final Bankroll: {row['Final_Bankroll']:.2f}\n"
                f"✅ Win Rate: {row['Win_Rate']:.2%}\n"
                f"💰 Avg EV: {row['Avg_EV']:.2f}\n"
                f"🎯 Avg Stake: {row['Avg_Stake']:.2f}\n"
                f"📊 Total Bets: {int(row['Total_Bets'])}\n\n"
            )
        except KeyError as e:
            raise DataError(f"Missing expected column in summary: {e}")

    # Trend analysis: find best model by final bankroll
    try:
        best_model = df.loc[df["Final_Bankroll"].idxmax()]
        trend_line = (
            f"📈 *Trend Analysis:* {best_model['Model']} outperformed others today "
            f"with a bankroll of {best_model['Final_Bankroll']:.2f}."
        )
        message += trend_line
    except Exception as e:
        logger.warning(f"⚠️ Trend analysis failed: {e}")

    # Send text summary
    send_message(message)

    # Send bankroll chart
    send_photo(chart_path, caption="📈 Bankroll Trajectories")


if __name__ == "__main__":
    main()