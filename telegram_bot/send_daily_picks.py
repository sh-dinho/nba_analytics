import os
import sqlite3
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import logging
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ======================
# Load config
# ======================
ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))

DB_PATH = CONFIG["database"]["path"]
TELEGRAM_TOKEN = CONFIG["notifications"]["telegram_token"]
CHAT_ID = CONFIG["notifications"]["telegram_chat_id"]
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ======================
# Telegram functions
# ======================
def send_message(text: str):
    url = f"{BASE_URL}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    requests.post(url, json=payload)


def send_photo(filepath: str, caption: str = ""):
    url = f"{BASE_URL}/sendPhoto"
    with open(filepath, "rb") as img:
        payload = {"chat_id": CHAT_ID, "caption": caption}
        files = {"photo": img}
        requests.post(url, data=payload, files=files)


# ======================
# Daily Picks & Graph
# ======================
def generate_bankroll_graph():
    """Create bankroll evolution plot."""
    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT Timestamp, CurrentBankroll FROM bankroll_tracker ORDER BY Timestamp", con)
        con.close()

        if df.empty:
            logging.warning("No bankroll data for graph.")
            return None

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        plt.figure(figsize=(8, 4))
        plt.plot(df["Timestamp"], df["CurrentBankroll"], linewidth=2, color="blue")
        plt.title("Bankroll Evolution")
        plt.xlabel("Date")
        plt.ylabel("Bankroll ($)")
        plt.grid(True)

        path = os.path.join(ROOT, "bankroll_graph.png")
        plt.savefig(path, dpi=150)
        plt.close()

        return path
    except Exception as e:
        logging.error(f"Error generating graph: {e}")
        return None


def send_daily_picks():
    """Send today's picks + bankroll graph."""
    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM daily_picks ORDER BY Timestamp DESC LIMIT 10", con)
        con.close()
    except Exception as e:
        logging.error(f"DB read error: {e}")
        return

    if df.empty:
        send_message("❌ No picks found today.")
        return

    # Format picks
    msg = "🏀 **TODAY'S PICKS**\n"
    today = datetime.now().strftime("%Y-%m-%d")

    for _, row in df.iterrows():
        msg += (
            f"\n\n🔹 *{row['Team']} vs {row['Opponent']}*"
            f"\n   📈 Prob: {row['Probability']:.2f}"
            f"\n   💰 Odds: {row['Odds']}"
            f"\n   📊 EV: {row['EV']:.3f}"
            f"\n   🧮 Stake: {row['SuggestedStake']:.2f}"
        )

    send_message(msg)

    # Add bankroll graph
    graph_path = generate_bankroll_graph()
    if graph_path:
        send_photo(graph_path, caption="📉 Bankroll Trend")
    else:
        send_message("⚠ Could not generate bankroll trend graph.")


# ======================
# Script entry point
# ======================
if __name__ == "__main__":
    logging.info("🚀 Sending daily picks...")
    send_daily_picks()
    logging.info("✔ Daily picks sent.")
