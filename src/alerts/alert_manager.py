from __future__ import annotations
# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Alert Manager (Upgraded with Batching)
# Author: Sadiq
# ============================================================

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from loguru import logger
from src.alerts.telegram import TelegramAlerts


class AlertManager:
    """
    Centralized alert router with batching, rate limiting,
    and severity-aware routing.

    Features:
      - Rate limiting per category
      - Batching for warnings/info
      - Immediate send for errors
      - High-level alert helpers
      - Generic alert API
    """

    def __init__(
        self,
        telegram: Optional[TelegramAlerts] = None,
        rate_limit_seconds: int = 60,
        batch_window_seconds: int = 300,  # 5 minutes
        enable_alerts: bool = True,
    ):
        self.telegram = telegram or TelegramAlerts()
        self.enable_alerts = enable_alerts and self.telegram.enabled

        self.rate_limit_seconds = rate_limit_seconds
        self.batch_window = timedelta(seconds=batch_window_seconds)

        # Track last alert timestamps by category
        self.last_sent: Dict[str, datetime] = {}

        # Batching buffers: category → list of messages
        self.batches: Dict[str, List[str]] = {}

        # Track when batch started
        self.batch_start: Dict[str, datetime] = {}

        if not self.enable_alerts:
            logger.warning("AlertManager: alerts disabled (Telegram not configured).")

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------
    def _can_send(self, category: str) -> bool:
        """Rate limiting for non-error categories."""
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

    def _send_now(self, text: str, markdown: bool = True):
        """Send immediately (used for errors or forced flush)."""
        if markdown:
            return self.telegram.send_markdown(text)
        return self.telegram.send_text(text)

    # --------------------------------------------------------
    # Batching logic
    # --------------------------------------------------------
    def _add_to_batch(self, category: str, message: str):
        """Add a message to a batch buffer."""
        if category not in self.batches:
            self.batches[category] = []
            self.batch_start[category] = datetime.utcnow()

        self.batches[category].append(message)

    def _should_flush(self, category: str) -> bool:
        """Check if batch window expired."""
        if category not in self.batch_start:
            return False

        return (datetime.utcnow() - self.batch_start[category]) >= self.batch_window

    def flush(self, category: Optional[str] = None):
        """
        Flush a specific category or all categories.
        """
        if not self.enable_alerts:
            return False

        if category:
            return self._flush_category(category)

        # Flush all
        for cat in list(self.batches.keys()):
            self._flush_category(cat)

    def _flush_category(self, category: str):
        """Flush a single category batch."""
        if category not in self.batches or not self.batches[category]:
            return False

        messages = self.batches[category]
        text = f"📢 *{category.upper()} — Batched Alerts*\n\n" + "\n".join(
            f"• {m}" for m in messages
        )

        logger.info(f"AlertManager: flushing {len(messages)} messages for '{category}'.")

        self._send_now(text)

        # Reset batch
        self.batches[category] = []
        self.batch_start[category] = datetime.utcnow()

        return True

    # --------------------------------------------------------
    # Public API — High-level alert types
    # --------------------------------------------------------
    def alert_pipeline_summary(self, summary: Dict[str, Any]):
        if not self._can_send("pipeline_summary"):
            return False
        return self.telegram.send_pipeline_summary(summary)

    def alert_error(self, context: str, error: str):
        """
        Errors bypass batching and rate limiting.
        """
        if not self.enable_alerts:
            return False

        text = f"❌ *ERROR in {context}*\n\n`{error}`"
        return self._send_now(text)

    def alert_data_quality(self, dq_report: Dict[str, Any]):
        msg = f"Data Quality Report:\n{dq_report}"
        return self.alert("data_quality", msg)

    def alert_model_monitor(self, monitor_report: Dict[str, Any]):
        msg = f"Model Monitor Report:\n{monitor_report}"
        return self.alert("model_monitor", msg)

    def alert_daily_betting_summary(self, summary: Dict[str, Any], target_date: str):
        msg = f"Daily Betting Summary ({target_date}):\n{summary}"
        return self.alert("daily_betting_summary", msg)

    # --------------------------------------------------------
    # Generic alert (with batching)
    # --------------------------------------------------------
    def alert(
        self,
        category: str,
        message: str,
        markdown: bool = True,
        severity: str = "info",  # "info" | "warning" | "error"
    ):
        """
        Generic alert with batching + severity routing.
        """

        # Errors bypass batching + rate limiting
        if severity == "error":
            text = f"❌ *{category.upper()} ERROR*\n\n{message}"
            return self._send_now(text, markdown=markdown)

        # Add to batch
        self._add_to_batch(category, message)

        # Flush if batch window expired
        if self._should_flush(category):
            return self._flush_category(category)

        # Otherwise, do nothing (batch will flush later)
        return True


if __name__ == "__main__":
    alerts = AlertManager()
    alerts.alert("test", "Batch message 1")
    alerts.alert("test", "Batch message 2")
    alerts.alert_error("test_context", "Immediate error")
    alerts.flush()  # manual flush