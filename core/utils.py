import requests
import logging


def send_telegram_message(msg: str):
    # Replace with your bot token and chat_id
    token = "YOUR_TELEGRAM_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg}
    try:
        requests.post(url, data=payload)
        logging.info("✔ Telegram message sent")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")