# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Central alert manager that routes pipeline,
#              monitoring, and betting alerts to Telegram.
# ============================================================

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from loguru import logger

from src.alerts.telegram import TelegramAlerts


class AlertManager:
    """
    Centralized alert router.

    Responsibilities:
      - Prevent alert spam (rate limiting)
      - Route alerts by severity
      - Provide high-level alert helpers for:
          * pipeline runs
          * errors
          * data quality reports
          * model monitor reports
          * daily betting summaries
      - Batch alerts when needed
    """

    def __init__(
        self,
        telegram: Optional[TelegramAlerts] = None,
        rate_limit_seconds: int = 60,
        enable_alerts: bool = True,
    ):
        self.telegram = telegram or TelegramAlerts()
        self.enable_alerts = enable_alerts and self.telegram.enabled
        self.rate_limit_seconds = rate_limit_seconds

        # Track last alert timestamps by category
        self.last_sent: Dict[str, datetime] = {}

        if not self.enable_alerts:
            logger.warning("AlertManager: alerts disabled (Telegram not configured).")

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------
    def _can_send(self, category: str) -> bool:
        """
        Rate limiting: ensure we don't spam alerts.
        """
        if not self.enable_alerts:
            return False

        now = datetime.utcnow()
        last = self.last_sent.get(category)

        if last is None:
            self.last_sent[category] = now
            return True

        if (now - last) >= timedelta(seconds=self.rate_limit_seconds):
            self.last_sent[category] = now
            return True

        logger.info(
            f"AlertManager: rate-limited alert '{category}' "
            f"(last sent {now - last} ago)."
        )
        return False

    def _send(self, category: str, text: str, markdown: bool = True):
        if not self._can_send(category):
            return False

        if markdown:
            return self.telegram.send_markdown(text)
        else:
            return self.telegram.send_text(text)

    # --------------------------------------------------------
    # Public API — High-level alert types
    # --------------------------------------------------------
    def alert_pipeline_summary(self, summary: Dict[str, Any]):
        """
        Send orchestrator run summary.
        """
        if not self._can_send("pipeline_summary"):
            return False

        return self.telegram.send_pipeline_summary(summary)

    def alert_error(self, context: str, error: str):
        """
        Send an error alert.
        """
        if not self._can_send("error"):
            return False

        return self.telegram.send_error(context, error)

    def alert_data_quality(self, dq_report: Dict[str, Any]):
        """
        Send data quality report.
        """
        if not self._can_send("data_quality"):
            return False

        return self.telegram.send_data_quality_report(dq_report)

    def alert_model_monitor(self, monitor_report: Dict[str, Any]):
        """
        Send model monitoring report.
        """
        if not self._can_send("model_monitor"):
            return False

        return self.telegram.send_model_monitor_report(monitor_report)

    def alert_daily_betting_summary(self, summary: Dict[str, Any], target_date: str):
        """
        Send daily betting summary.
        """
        if not self._can_send("daily_betting_summary"):
            return False

        return self.telegram.send_daily_betting_summary(summary, target_date)

    # --------------------------------------------------------
    # Generic alert
    # --------------------------------------------------------
    def alert(
        self,
        category: str,
        message: str,
        markdown: bool = True,
    ):
        """
        Generic alert for custom categories.
        """
        if not self._can_send(category):
            return False

        return self._send(category, message, markdown=markdown)


if __name__ == "__main__":
    alerts = AlertManager()
    alerts.alert("test", "*AlertManager test message* — everything wired correctly.")
