# core/utils.py (Updated)
import requests
import logging
from config import TEAM_MAP # Import the map

def get_standardized_team_name(api_name: str) -> str:
    """Standardize the team name using the predefined map, defaulting to the original name."""
    return TEAM_MAP.get(api_name, api_name)


def send_telegram_message(msg: str):
    # Replace with your bot token and chat_id (or use ENV)
    token = "8494881512:AAEzEh3NE-T4YnVO6zR-uW-lTowIFKy1o50"
    chat_id = "7879485351"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg}
    try:
        requests.post(url, data=payload)
        logging.info("✔ Telegram message sent")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")