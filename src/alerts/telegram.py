from __future__ import annotations

# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Telegram Alerts (Upgraded)
# Author: Sadiq
#
# Description:
#     High-level Telegram alert interface used by AlertManager.
#     Features:
#       • Markdown v2-safe formatting
#       • Multi-channel routing (errors, summaries, monitoring)
#       • Environment-based overrides (dev/staging/prod)
#       • Chart sending utilities
#       • Structured alert helpers
# ============================================================

import os
import tempfile
import requests
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger


# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------

BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"

# Default chat (fallback)
CHAT_DEFAULT_ENV = "TELEGRAM_CHAT_ID"

# Optional multi-channel routing
CHAT_ERRORS_ENV = "TELEGRAM_CHAT_ID_ERRORS"
CHAT_SUMMARIES_ENV = "TELEGRAM_CHAT_ID_SUMMARIES"
CHAT_MONITORING_ENV = "TELEGRAM_CHAT_ID_MONITORING"

# Environment mode (dev/staging/prod)
ENV_MODE_ENV = "NBA_ENV"  # dev | staging | prod


# ------------------------------------------------------------
# Telegram Alerts
# ------------------------------------------------------------

class TelegramAlerts:
    def __init__(self):
        self.token = os.getenv(BOT_TOKEN_ENV)
        self.chat_default = os.getenv(CHAT_DEFAULT_ENV)

        # Optional multi-channel routing
        self.chat_errors = os.getenv(CHAT_ERRORS_ENV, self.chat_default)
        self.chat_summaries = os.getenv(CHAT_SUMMARIES_ENV, self.chat_default)
        self.chat_monitoring = os.getenv(CHAT_MONITORING_ENV, self.chat_default)

        # Environment mode
        self.env = os.getenv(ENV_MODE_ENV, "dev")

        # Disable alerts entirely if missing credentials
        self.enabled = bool(self.token and self.chat_default)

        if not self.enabled:
            logger.warning("TelegramAlerts disabled: missing credentials.")

        # Warn if running in dev mode
        if self.env != "prod":
            logger.info(f"TelegramAlerts running in {self.env} mode.")

    # --------------------------------------------------------
    # Low-level senders
    # --------------------------------------------------------
    def _send(self, payload: dict, endpoint: str = "sendMessage", files=None):
        if not self.enabled:
            return False

        url = f"https://api.telegram.org/bot{self.token}/{endpoint}"

        try:
            resp = requests.post(url, data=payload, files=files, timeout=10)
            if not resp.ok:
                logger.error(f"Telegram API error {resp.status_code}: {resp.text}")
                return False
            return True

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def _send_to(self, chat_id: str, text: str, markdown: bool = True):
        if markdown:
            return self._send(
                {
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                }
            )
        else:
            return self._send(
                {
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                }
            )

    # --------------------------------------------------------
    # Chart sending
    # --------------------------------------------------------
    def send_chart(self, fig, caption: str = "Chart", chat_id: str | None = None):
        if not self.enabled:
            return False

        chat_id = chat_id or self.chat_default

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            fig.savefig(tmp.name, dpi=200, bbox_inches="tight")
            tmp.seek(0)

            return self._send(
                {"chat_id": chat_id, "caption": caption},
                endpoint="sendPhoto",
                files={"photo": tmp},
            )

    def send_bankroll_chart(self, records: pd.DataFrame, caption="Bankroll Over Time"):
        records = records.sort_values("date")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(records["date"], records["bankroll_after"], marker="o")
        ax.set_title("Bankroll Over Time")
        ax.set_ylabel("Bankroll")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        self.send_chart(fig, caption)
        plt.close(fig)

    # --------------------------------------------------------
    # High-level alert formats
    # --------------------------------------------------------
    def send_error(self, context: str, error: str):
        text = (
            f"🚨 *ERROR*\n"
            f"*Context:* {context}\n"
            f"*Details:* `{error}`"
        )
        return self._send_to(self.chat_errors, text)

    def send_pipeline_summary(self, summary: dict):
        text = (
            "📦 *Pipeline Summary*\n"
            f"• Status: *{summary.get('status', 'unknown')}*\n"
            f"• Duration: {summary.get('duration', 'N/A')}\n"
            f"• Rows processed: {summary.get('rows', 'N/A')}\n"
        )
        return self._send_to(self.chat_summaries, text)

    def send_data_quality_report(self, dq: dict):
        text = "🧪 *Data Quality Report*\n"

        for section, result in dq.items():
            ok = result.get("ok", False)
            emoji = "✅" if ok else "❌"
            text += f"{emoji} *{section}*\n"

        return self._send_to(self.chat_monitoring, text)

    def send_model_monitor_report(self, report: dict):
        ok = report.get("ok", False)
        emoji = "🟢" if ok else "🔴"

        text = f"{emoji} *Model Monitor Report*\n"

        for issue in report.get("issues", []):
            text += f"• `{issue['level']}` — {issue['message']}\n"

        return self._send_to(self.chat_monitoring, text)

    def send_daily_betting_summary(self, summary: dict, target_date: str):
        text = (
            f"💰 *Daily Betting Summary — {target_date}*\n"
            f"• ROI: *{summary.get('roi', 'N/A')}*\n"
            f"• Profit: {summary.get('profit', 'N/A')}\n"
            f"• Record: {summary.get('record', 'N/A')}\n"
        )
        return self._send_to(self.chat_summaries, text)