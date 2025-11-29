import requests
import yaml
import logging
import sqlite3
import pandas as pd

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
TELEGRAM_TOKEN = CONFIG["notifications"]["telegram_token"]
TELEGRAM_CHAT_ID = int(CONFIG["notifications"]["telegram_chat_id"])
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_message(text: str):
    url = f"{BASE_URL}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        requests.post(url, json=payload, timeout=10)
        logging.info("✅ Telegram notification sent")
    except Exception as e:
        logging.error(f"❌ Failed to send Telegram message: {e}")

def send_daily_picks(df: pd.DataFrame):
    if df.empty:
        send_message("❌ No positive EV picks found today.")
        return

    con = sqlite3.connect(DB_PATH)
    tracker = pd.read_sql("SELECT * FROM bankroll_tracker ORDER BY Timestamp DESC LIMIT 1", con)
    con.close()

    bankroll_msg = ""
    if not tracker.empty:
        row = tracker.iloc[-1]
        bankroll_msg = f"\n💰 Bankroll: ${row['CurrentBankroll']:.2f} | ROI: {row['ROI']:.2%}"

    msg = "💰 --- TODAY'S VALUE BETS --- 💰\n"
    for _, row in df.iterrows():
        msg += f"\n{row['Team']} vs {row['Opponent']} | Odds {row['Odds']} | Prob {row['Probability']:.2f} | EV {row['EV']:.3f} | Stake {row['SuggestedStake']:.2f}"
    msg += bankroll_msg
    send_message(msg)
