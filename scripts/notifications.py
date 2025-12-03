# ============================================================
# File: notifications.py
# Purpose: Send pipeline notifications (Telegram, summaries)
# ============================================================

import os
import requests
import pandas as pd

# Load Telegram credentials from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_message(msg: str) -> None:
    """
    Send a plain text message to Telegram.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials not set. Skipping notification.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        if resp.status_code != 200:
            print(f"❌ Failed to send Telegram message: {resp.text}")
        else:
            print("✅ Telegram message sent successfully")
    except Exception as e:
        print(f"❌ Error sending Telegram message: {e}")


def send_ev_summary(picks: pd.DataFrame) -> None:
    """
    Send an EV (expected value) summary of picks to Telegram.
    Expects a DataFrame with at least 'pick' and 'ev' columns.
    """
    if picks is None or picks.empty:
        send_telegram_message("No picks available for EV summary.")
        return

    summary_lines = []
    if "pick" in picks.columns and "ev" in picks.columns:
        grouped = picks.groupby("pick")["ev"].mean().reset_index()
        for _, row in grouped.iterrows():
            summary_lines.append(f"{row['pick']}: avg EV={row['ev']:.3f}")
    else:
        summary_lines.append("⚠️ Picks DataFrame missing 'pick' or 'ev' columns.")

    msg = "EV Summary:\n" + "\n".join(summary_lines)
    send_telegram_message(msg)