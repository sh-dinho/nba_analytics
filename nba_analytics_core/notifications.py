# File: nba_analytics_core/notifications.py

import os
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(text: str):
    """
    Send a message to Telegram chat using bot API.
    Requires TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in environment.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set in environment.")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Telegram error: {e}")
        return False
    return True


def send_ev_summary(df_odds):
    """
    Send a summary of positive EV bets to Telegram.
    """
    if df_odds.empty:
        return False

    num_positive_ev = ((df_odds["ev_home"] > 0) | (df_odds["ev_away"] > 0)).sum()

    best_home = df_odds.loc[df_odds["ev_home"].idxmax()] if not df_odds["ev_home"].isna().all() else None
    best_away = df_odds.loc[df_odds["ev_away"].idxmax()] if not df_odds["ev_away"].isna().all() else None

    msg = f"📊 Positive EV Bets Today: {num_positive_ev}\n"
    if best_home is not None:
        msg += f"🏠 Best Home EV: {best_home['home_team']} vs {best_home['away_team']} (+{best_home['ev_home']:.2f})\n"
    if best_away is not None:
        msg += f"🛫 Best Away EV: {best_away['away_team']} @ {best_away['home_team']} (+{best_away['ev_away']:.2f})"

    return send_telegram_message(msg)