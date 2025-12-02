# ============================================================
# File: scripts/telegram_report.py
# Purpose: Send bankroll summary + chart + trend analysis to Telegram
# ============================================================

import os
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set. Skipping report.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
        logger.info("📲 Telegram text report sent successfully")
    except Exception as e:
        logger.error(f"❌ Failed to send Telegram text report: {e}")

def send_photo(photo_path: str, caption: str = None):
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
            requests.post(url, data=data, files=files)
            logger.info("📸 Telegram chart sent successfully")
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram chart: {e}")

def main():
    # Load combined summary
    if not os.path.exists("results/combined_summary.csv"):
        logger.warning("⚠️ No combined_summary.csv found. Skipping report.")
        return

    df = pd.read_csv("results/combined_summary.csv")

    # Format summary message
    message = "*🏀 NBA Daily Combined Report*\n\n"
    for _, row in df.iterrows():
        message += (
            f"📌 Model: {row['Model']}\n"
            f"🏦 Final Bankroll: {row['Final_Bankroll']:.2f}\n"
            f"✅ Win Rate: {row['Win_Rate']:.2%}\n"
            f"💰 Avg EV: {row['Avg_EV']:.2f}\n"
            f"🎯 Avg Stake: {row['Avg_Stake']:.2f}\n"
            f"📊 Total Bets: {int(row['Total_Bets'])}\n\n"
        )

    # Trend analysis: find best model by final bankroll
    best_model = df.loc[df["Final_Bankroll"].idxmax()]
    trend_line = f"📈 *Trend Analysis:* {best_model['Model']} outperformed others today with a bankroll of {best_model['Final_Bankroll']:.2f}."

    message += trend_line

    # Send text summary
    send_message(message)

    # Send bankroll chart
    send_photo("results/bankroll_comparison.png", caption="📈 Bankroll Trajectories")

if __name__ == "__main__":
    main()