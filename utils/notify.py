import requests
import yaml
import sqlite3
import logging
import pandas as pd
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
CONFIG = yaml.safe_load(open(CONFIG_PATH))

TELEGRAM_TOKEN = CONFIG["notifications"]["telegram_token"]
TELEGRAM_CHAT_ID = CONFIG["notifications"]["telegram_chat_id"]
DB_PATH = CONFIG["database"]["path"]

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_telegram_message(text: str):
    try:
        requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text
        })
        logging.info("✔ Telegram message sent")
    except Exception as e:
        logging.error(f"❌ Failed to send message: {e}")

def send_daily_picks(df: pd.DataFrame):
    if df.empty:
        send_telegram_message("❌ No picks today.")
        return

    con = sqlite3.connect(DB_PATH)
    tracker = pd.read_sql("SELECT * FROM bankroll_tracker ORDER BY Timestamp DESC LIMIT 1", con)
    con.close()

    bankroll_msg = ""
    if not tracker.empty:
        row = tracker.iloc[0]
        bankroll_msg = f"\n💰 Bankroll: ${row['CurrentBankroll']:.2f} | ROI: {row['ROI']:.2%}"

    msg = "💰 --- TODAY'S VALUE BETS --- 💰\n"
    for _, row in df.iterrows():
        msg += f"\n{row['Team']} vs {row['Opponent']} | Odds {row['Odds']} | Prob {row['Probability']:.2f} | EV {row['EV']:.3f} | Stake {row['SuggestedStake']:.2f}"
    msg += bankroll_msg
    send_telegram_message(msg)
