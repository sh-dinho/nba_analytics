# ============================================================
# 🏀 NBA Analytics v3
# Module: Telegram Alerts — Betting Recommendations
# File: src/alerts/recommendations_alert.py
# Author: Sadiq
#
# Description:
#     Sends daily betting recommendations to Telegram.
# ============================================================

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.alerts.telegram import send_telegram_message


def format_recommendations_message(df: pd.DataFrame, pred_date):
    lines = [
        f"🏀 *NBA Betting Recommendations*",
        f"Date: `{pred_date}`",
        "",
    ]

    for _, r in df.iterrows():
        line = (
            f"*{r['market']}* — {r['team']}\n"
            f"→ *{r['recommendation']}*\n"
            f"Confidence: `{r['confidence']}`\n"
            f"Edge: `{r['edge']}`"
        )
        if r["risk_flags"]:
            line += f"\n⚠️ {r['risk_flags']}"
        lines.append(line)
        lines.append("")

    return "\n".join(lines)


def send_recommendations_alert(df: pd.DataFrame, pred_date):
    if df.empty:
        logger.warning("No recommendations to send.")
        return

    message = format_recommendations_message(df, pred_date)

    try:
        send_telegram_message(message)
        logger.success("📨 Recommendations sent to Telegram.")
    except Exception as e:
        logger.error(f"Failed to send recommendations alert: {e}")
