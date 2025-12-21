# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Telegram Alerts
# Author: Sadiq
#
# Description:
#     Utilities for sending alerts and charts to Telegram:
#       - Text alerts (summary, value bets, errors)
#       - Bankroll charts via matplotlib figures
# ============================================================

from __future__ import annotations

import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import requests


TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"


def _get_credentials() -> tuple[str, str]:
    token = os.getenv(TELEGRAM_BOT_TOKEN_ENV)
    chat_id = os.getenv(TELEGRAM_CHAT_ID_ENV)

    if not token:
        raise RuntimeError(
            f"Missing Telegram bot token. Set {TELEGRAM_BOT_TOKEN_ENV} in your environment."
        )
    if not chat_id:
        raise RuntimeError(
            f"Missing Telegram chat ID. Set {TELEGRAM_CHAT_ID_ENV} in your environment."
        )

    return token, chat_id


def send_telegram_message(text: str) -> None:
    token, chat_id = _get_credentials()
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    resp = requests.post(url, json=payload, timeout=10)
    if not resp.ok:
        raise RuntimeError(f"Telegram API error {resp.status_code}: {resp.text}")


def _send_telegram_chart(fig, caption: str = "Chart") -> None:
    token, chat_id = _get_credentials()
    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        fig.savefig(tmp.name, dpi=200, bbox_inches="tight")
        tmp.seek(0)
        files = {"photo": tmp}
        data = {"chat_id": chat_id, "caption": caption}
        resp = requests.post(url, data=data, files=files, timeout=10)
        if not resp.ok:
            raise RuntimeError(f"Telegram API error {resp.status_code}: {resp.text}")


def send_bankroll_chart(
    records: pd.DataFrame, caption: str = "Bankroll Over Time"
) -> None:
    records = records.sort_values("date")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(records["date"], records["bankroll_after"], marker="o")
    ax.set_title("Bankroll Over Time")
    ax.set_ylabel("Bankroll")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    _send_telegram_chart(fig, caption=caption)
    plt.close(fig)
