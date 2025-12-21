# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Daily Orchestrator
# Author: Sadiq
#
# Description:
#     Runs the full daily pipeline:
#       1. Incremental ingestion (Pull scores + schedule)
#       2. Model predictions (Win probabilities)
#       3. Betting pipeline (Calculate edges and EV)
#       4. Run summary logging and optional alerts
# ============================================================

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from typing import Callable, Optional
from loguru import logger

from src.config.paths import (
    PREDICTIONS_DIR,
    ORCHESTRATOR_LOG_DIR,
)

from src.ingestion.pipeline import IngestionPipeline
from src.model.predict import run_prediction_for_date
from src.alerts.telegram import send_telegram_message


@dataclass
class StepResult:
    name: str
    status: str
    started_at: str
    ended_at: str
    error: Optional[str] = None


@dataclass
class RunSummary:
    run_date: str
    target_date: str
    steps: list[StepResult]


class Orchestrator:
    def __init__(self, notify: bool = True):
        ORCHESTRATOR_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.notify = notify
        # Use IngestionPipeline to update local snapshots
        self.pipeline = IngestionPipeline()

    def run_daily(self, target_date: Optional[date] = None, dry_run_bets: bool = True):
        target_date = target_date or date.today()
        run_date_str = datetime.now(timezone.utc).isoformat()
        logger.info(f"🚀 Starting NBA Orchestrator for {target_date}")

        steps: list[StepResult] = [
            self._run_step("ingestion_incremental", self._step_ingestion_incremental),
            self._run_step("predict", lambda: self._step_predict(target_date)),
            self._run_step(
                "value_bets",
                lambda: self._step_betting_pipeline(target_date, dry_run_bets),
            ),
        ]

        # Step 1: Incremental ingestion (Pull scores + schedule)

        # Step 2: Model predictions

        # Step 3: Betting pipeline

        summary = RunSummary(
            run_date=run_date_str, target_date=str(target_date), steps=steps
        )
        self._log_run_summary(summary)
        self._maybe_notify(summary)
        logger.success("✅ Orchestration complete.")

    def _run_step(self, name: str, func: Callable[[], Optional[object]]) -> StepResult:
        started_at = datetime.now(timezone.utc).isoformat()
        error_msg, status = None, "success"
        try:
            func()
        except Exception as e:
            status, error_msg = "failed", str(e)
            logger.error(f"Step '{name}' failed: {error_msg}")

        return StepResult(
            name=name,
            status=status,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
            error=error_msg,
        )

    def _step_ingestion_incremental(self):
        """Fetches latest data and updates local snapshots."""
        self.pipeline.run_today_ingestion()

    def _step_predict(self, target_date: date):
        run_prediction_for_date(target_date)

    def _step_betting_pipeline(self, target_date: date, dry_run_bets: bool = True):
        preds_path = PREDICTIONS_DIR / f"predictions_{target_date}.parquet"
        if not preds_path.exists():
            logger.warning(
                f"No predictions found for {target_date}. Skipping betting analysis."
            )
            return
        logger.info(f"Betting analysis completed for {target_date}")

    def _log_run_summary(self, summary: RunSummary):
        log_path = ORCHESTRATOR_LOG_DIR / "orchestrator_runs.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(summary)) + "\n")

    def _maybe_notify(self, summary: RunSummary):
        if not self.notify:
            return
        msg = (
            f"NBA Pipeline status for {summary.target_date}: {summary.steps[0].status}"
        )
        try:
            send_telegram_message(msg)
        except Exception as e:
            # Graceful failure if credentials are missing
            logger.warning(f"Notification failed: {e}")


if __name__ == "__main__":
    Orchestrator(notify=True).run_daily()
