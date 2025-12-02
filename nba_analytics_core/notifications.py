# Path: nba_analytics_core/notifications.py

import os
import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(text: str, chat_id: str = None, token: str = None, parse_mode: str = None) -> bool:
    """
    Send a message to Telegram chat using bot API.
    Requires TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in environment unless overridden.
    """
    token = token or TELEGRAM_TOKEN
    chat_id = chat_id or TELEGRAM_CHAT_ID

    if not token or not chat_id:
        logger.error("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set in environment.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("✅ Telegram message sent successfully")
        return True
    except requests.RequestException as e:
        logger.error(f"Telegram error: {e}")
        return False


def send_ev_summary(df_odds) -> bool:
    """
    Send a summary of positive EV bets to Telegram.
    Expects df_odds with columns: ev_home, ev_away, home_team, away_team.
    """
    required_cols = {"ev_home", "ev_away", "home_team", "away_team"}
    if df_odds.empty or not required_cols.issubset(df_odds.columns):
        logger.warning("EV summary skipped: missing data or required columns.")
        return False

    num_positive_ev = ((df_odds["ev_home"] > 0) | (df_odds["ev_away"] > 0)).sum()

    best_home = None
    best_away = None
    if not df_odds["ev_home"].isna().all():
        best_home = df_odds.loc[df_odds["ev_home"].idxmax()]
    if not df_odds["ev_away"].isna().all():
        best_away = df_odds.loc[df_odds["ev_away"].idxmax()]

    msg = f"📊 Positive EV Bets Today: {num_positive_ev}\n"
    if best_home is not None:
        msg += f"🏠 Best Home EV: {best_home['home_team']} vs {best_home['away_team']} (+{best_home['ev_home']:.2f})\n"
    if best_away is not None:
        msg += f"🛫 Best Away EV: {best_away['away_team']} @ {best_away['home_team']} (+{best_away['ev_away']:.2f})"

    return send_telegram_message(msg)


def send_bankroll_summary(metrics: dict) -> bool:
    """
    Send bankroll metrics summary to Telegram.
    Expects metrics dict with keys: final_bankroll, roi, win_rate, wins, losses.
    """
    if not metrics:
        logger.warning("Bankroll summary skipped: no metrics provided.")
        return False

    msg = (
        f"💰 Bankroll Summary\n"
        f"Final Bankroll: ${metrics.get('final_bankroll', 0):.2f}\n"
        f"ROI: {metrics.get('roi', 0)*100:.2f}%\n"
        f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}% "
        f"({metrics.get('wins', 0)}W/{metrics.get('losses', 0)}L)"
    )
    return send_telegram_message(msg)