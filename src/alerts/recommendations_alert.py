from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Telegram Alerts — Betting Recommendations
# File: src/alerts/recommendations_alert.py
# Author: Sadiq
#
# Description:
#     Sends daily betting recommendations to Telegram using the
#     upgraded AlertManager (with batching + severity routing).
# ============================================================

import pandas as pd
from loguru import logger

from src.alerts.alert_manager import AlertManager


# ------------------------------------------------------------
# Message Formatting
# ------------------------------------------------------------

def format_recommendations_message(df: pd.DataFrame, pred_date: str) -> str:
    """
    Build a clean Markdown message for Telegram.
    """
    lines = [
        "🏀 *NBA Betting Recommendations*",
        f"📅 Date: `{pred_date}`",
        "",
    ]

    for _, r in df.iterrows():
        block = [
            f"*{r['market']}* — *{r['team']}*",
            f"→ *{r['recommendation']}*",
            f"Confidence: `{r['confidence']}`",
            f"Edge: `{r['edge']}`",
        ]

        risk = r.get("risk_flags")
        if risk:
            block.append(f"⚠️ {risk}")

        lines.append("\n".join(block))
        lines.append("")  # spacing

    return "\n".join(lines).strip()


# ------------------------------------------------------------
# Alert Sender
# ------------------------------------------------------------

def send_recommendations_alert(df: pd.DataFrame, pred_date: str) -> bool:
    """
    Sends the formatted recommendations via AlertManager.
    Uses batching under the hood (category="betting_recommendations").
    """
    if df.empty:
        logger.warning("No recommendations to send.")
        return False

    message = format_recommendations_message(df, pred_date)
    alerts = AlertManager()

    try:
        alerts.alert(
            category="betting_recommendations",
            message=message,
            markdown=True,
            severity="info",  # batched by default
        )
        logger.success("📨 Betting recommendations queued for Telegram.")
        return True

    except Exception as e:
        logger.error(f"Failed to send recommendations alert: {e}")
        return False