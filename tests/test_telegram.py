
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

token = os.getenv("TELEGRAM_BOT_TOKEN")
channel = os.getenv("TELEGRAM_CHAT_ID")

msg = "🔔 Test notification from .env setup!"

resp = requests.post(
    f"https://api.telegram.org/bot{token}/sendMessage",
    data={"chat_id": channel, "text": msg}
)

print("Status:", resp.status_code)
print("Response:", resp.json())

# import os, requests
# from dotenv import load_dotenv

# load_dotenv()
# token = os.getenv("TELEGRAM_BOT_TOKEN")

# resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates")
# print(resp.json())