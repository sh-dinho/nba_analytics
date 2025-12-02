# ============================================================
# File: scripts/telegram_report.py
# Purpose: Send bankroll summary to Telegram
# ============================================================

import os
import logging
import requests
import pandas as pd
from scripts.Utils import Simulation   # <-- NEW

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
        logger.info("📲 Telegram report sent successfully")
    except Exception as e:
        logger.error(f"❌ Failed to send Telegram report: {e}")

def main():
    # Load picks_bankroll.csv
    if not os.path.exists("results/picks_bankroll.csv"):
        logger.warning("⚠️ No picks_bankroll.csv found. Skipping report.")
        return

    df = pd.read_csv("results/picks_bankroll.csv")

    # Reconstruct simulation from history
    sim = Simulation(initial_bankroll=df.iloc[0]["bankroll"])
    for _, row in df.iterrows():
        sim.history.append(row.to_dict())
    sim.bankroll = df.iloc[-1]["bankroll"]

    summary = sim.summary()

    # Format message
    message = (
        f"🏀 *NBA Daily Report*\n\n"
        f"🏦 Final Bankroll: {summary['Final_Bankroll']:.2f}\n"
        f"✅ Win Rate: {summary['Win_Rate']:.2%}\n"
        f"💰 Avg EV: {summary['Avg_EV']:.2f}\n"
        f"🎯 Avg Stake: {summary['Avg_Stake']:.2f}\n"
        f"📊 Total Bets: {summary['Total_Bets']}"
    )

    send_message(message)

if __name__ == "__main__":
    main()