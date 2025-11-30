import logging
import requests
import yaml

logging.basicConfig(level=logging.INFO)

# --- Load config safely ---
try:
    with open("config.yaml") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    logging.warning("⚠️ config.yaml not found, using defaults.")
    CONFIG = {"telegram": {"bot_token": "", "chat_id": ""}}

TELEGRAM_TOKEN = CONFIG.get("telegram", {}).get("bot_token", "")
TELEGRAM_CHAT_ID = CONFIG.get("telegram", {}).get("chat_id", "")

def send_telegram_message(message: str):
    """
    Send a message to Telegram using bot token and chat ID.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram not configured. Skipping message.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        resp = requests.post(url, json=payload)
        if resp.status_code == 200:
            logging.info("✔ Telegram message sent")
        else:
            logging.error(f"❌ Failed to send Telegram message: {resp.text}")
    except Exception as e:
        logging.error(f"❌ Error sending Telegram message: {e}")

def format_bankroll_update(bankroll: float, roi: float) -> str:
    """
    Format bankroll update message for Telegram.
    """
    return f"Bankroll update:\n💰 Current bankroll: ${bankroll:.2f}\n📈 ROI: {roi:.2%}"

def format_pick(home_team: str, away_team: str, ev: float) -> str:
    """
    Format daily pick message for Telegram.
    """
    return f"Today's pick:\n🏀 {home_team} vs {away_team}\nExpected Value: {ev:.2f}"