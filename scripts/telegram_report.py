# ============================================================
# File: scripts/telegram_report.py
# Purpose: Read picks_bankroll.csv and send bankroll, win rate, EV, and Kelly metrics to Telegram
# ============================================================

import pandas as pd
import requests
import os

# Telegram bot config
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_message(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

def ev_emoji(ev_value):
    """Return emoji based on EV sign."""
    if pd.isna(ev_value):
        return "⚪"
    if ev_value > 0:
        return "🟢"
    elif ev_value < 0:
        return "🔴"
    else:
        return "⚪"

def bankroll_emoji(current, previous):
    """Return emoji based on bankroll trajectory."""
    if previous is None or pd.isna(previous):
        return "⚪"
    if current > previous:
        return "📈"
    elif current < previous:
        return "📉"
    else:
        return "⚪"

def build_report(csv_path="results/picks_bankroll.csv"):
    df = pd.read_csv(csv_path)

    # Last row is the summary
    summary = df.tail(1).to_dict(orient="records")[0]

    # Build header with emojis + bold
    report_lines = [
        "*📊 Daily Betting Report*",
        f"🏦 *Final Bankroll:* {summary.get('Final_Bankroll', 'N/A')}",
        f"✅ *Win Rate:* {summary.get('Win_Rate', 'N/A')}",
        f"💰 *Avg EV:* {summary.get('Avg_EV', 'N/A')}",
        f"🎯 *Avg Kelly Bet:* {summary.get('Avg_Kelly_Bet', 'N/A')}",
        "",
        "*🎮 Game Details:*"
    ]

    # Add per-game details (exclude summary row)
    prev_bankroll = None
    for _, row in df.iloc[:-1].iterrows():
        ev_val = row.get("EV", None)
        emoji_ev = ev_emoji(ev_val)
        emoji_bankroll = bankroll_emoji(row.get("bankroll", None), prev_bankroll)

        line = (
            f"{emoji_ev}{emoji_bankroll} {row['home_team']} vs {row['away_team']} → "
            f"*Winner:* {row['winner']} "
            f"(Conf: {row['winner_confidence']}%)\n"
            f"   EV: {ev_val} | Kelly: {row.get('Kelly_Bet', 'N/A')} | Bankroll: {row.get('bankroll', 'N/A')}"
        )
        report_lines.append(line)
        prev_bankroll = row.get("bankroll", None)

    return "\n".join(report_lines)

if __name__ == "__main__":
    report = build_report()
    send_message(report)