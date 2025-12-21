"""
Daily Orchestrator
------------------
Runs the full daily pipeline:

1. Incremental ingestion (or reuse existing snapshots)
2. Model predictions for the target date
3. Value bets and recommendations
4. (Optional) Auto-betting
5. Run summary logging and notifications
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from loguru import logger

from src.config.paths import (
    SCHEDULE_SNAPSHOT,
    LONG_SNAPSHOT,
    ODDS_DIR,
    PREDICTIONS_DIR,
    ORCHESTRATOR_LOG_DIR,
)
from src.model.predict import run_prediction_for_date
from src.alerts.telegram import send_telegram_message  # we'll define this


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

    # -----------------------------------------------------
    # Public entry point
    # -----------------------------------------------------

    def run_daily(self, target_date: Optional[date] = None, dry_run_bets: bool = True):
        target_date = target_date or date.today()
        run_date_str = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"=== Orchestrator: run_daily start (target_date={target_date}) ==="
        )

        steps: list[StepResult] = []

        # 1) Incremental ingestion (or reuse snapshots)
        steps.append(
            self._run_step(
                "ingestion_incremental",
                self._step_ingestion_incremental,
            )
        )

        # 2) Predictions
        steps.append(
            self._run_step(
                "predict",
                lambda: self._step_predict(target_date),
            )
        )

        # 3) Betting pipeline (value bets + recommendations)
        steps.append(
            self._run_step(
                "value_bets_and_recommendations",
                lambda: self._step_betting_pipeline(target_date, dry_run_bets),
            )
        )

        summary = RunSummary(
            run_date=run_date_str,
            target_date=str(target_date),
            steps=steps,
        )

        self._log_run_summary(summary)
        self._maybe_notify(summary)

        logger.info("=== Orchestrator: run_daily complete ===")

    # -----------------------------------------------------
    # Step wrapper with timing and error handling
    # -----------------------------------------------------

    def _run_step(self, name: str, func: Callable[[], Optional[object]]) -> StepResult:
        started_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Orchestrator: starting step '{name}'")

        error_msg: Optional[str] = None
        status = "success"

        try:
            func()
        except Exception as e:
            status = "failed"
            error_msg = str(e)
            logger.error(f"Orchestrator: step '{name}' failed with error: {error_msg}")
        else:
            logger.info(f"Orchestrator: step '{name}' completed successfully.")

        ended_at = datetime.now(timezone.utc).isoformat()

        return StepResult(
            name=name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            error=error_msg,
        )

    # -----------------------------------------------------
    # Step implementations
    # -----------------------------------------------------

    def _step_ingestion_incremental(self):
        """
        For now, we just confirm snapshots exist and log their sizes.
        This matches the behavior you saw: "using existing snapshots".
        """
        if not SCHEDULE_SNAPSHOT.exists() or not LONG_SNAPSHOT.exists():
            raise FileNotFoundError(
                "Schedule or long-format snapshots missing. Run full ingestion first."
            )

        df_sched = pd.read_parquet(SCHEDULE_SNAPSHOT)
        df_long = pd.read_parquet(LONG_SNAPSHOT)

        logger.info(
            f"_step_ingestion_incremental(): using existing snapshots "
            f"({len(df_sched)} games, {len(df_long)} long rows)."
        )

    def _step_predict(self, target_date: date):
        run_prediction_for_date(target_date)

    def _step_betting_pipeline(self, target_date: date, dry_run_bets: bool = True):
        """
        Loads odds and predictions for the target date,
        computes value bets, and (optionally) simulates or places bets.
        """
        # Load odds
        odds_path = ODDS_DIR / f"odds_{target_date}.parquet"
        if not odds_path.exists():
            raise FileNotFoundError(
                f"Odds snapshot not found for date={target_date} at {odds_path}. "
                f"Fetch and save odds before running betting pipeline."
            )

        odds_df = pd.read_parquet(odds_path)
        logger.info(
            f"Loaded {len(odds_df)} odds rows for date={target_date} from {odds_path}."
        )

        # Find predictions file
        preds_path = self._find_predictions_for_date(target_date)
        preds_df = pd.read_parquet(preds_path)

        # Example: simple join on game_id + team
        merged = odds_df.merge(
            preds_df,
            on=["game_id", "team"],
            how="inner",
            suffixes=("_odds", "_pred"),
        )

        if "win_probability" not in merged.columns:
            raise KeyError("win_probability column missing in predictions.")

        logger.info(
            f"Joined odds + predictions for {len(merged)} rows. Dry run bets = {dry_run_bets}"
        )

        # TODO: implement actual edge + stake logic here
        # For now we just log the top edges:
        if "price" in merged.columns:
            # Placeholder: log top few rows
            logger.info(
                "Sample merged odds+predictions:\n{}", merged.head().to_string()
            )

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def _find_predictions_for_date(self, target_date: date) -> Path:
        preds_path = PREDICTIONS_DIR / f"predictions_{target_date}.parquet"
        if not preds_path.exists():
            raise FileNotFoundError(
                f"No predictions file found for date={target_date} in {PREDICTIONS_DIR}."
            )
        return preds_path

    def _log_run_summary(self, summary: RunSummary):
        log_path = ORCHESTRATOR_LOG_DIR / "orchestrator_runs.log"
        records = []

        if log_path.exists():
            existing = log_path.read_text().strip()
            if existing:
                for line in existing.splitlines():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        record = asdict(summary)
        records.append(record)

        with log_path.open("w", encoding="utf-8") as f:
            for r in records:
                line = json.dumps(r)
                f.write(line + "\n")

    def _maybe_notify(self, summary: RunSummary):
        if not self.notify:
            return

        # Simple text summary
        lines = [
            f"NBA Orchestrator run",
            f"Run date (UTC): {summary.run_date}",
            f"Target date: {summary.target_date}",
        ]
        for step in summary.steps:
            lines.append(
                f"- {step.name}: {step.status}"
                + (f" (error: {step.error})" if step.error else "")
            )

        message = "\n".join(lines)

        try:
            send_telegram_message(message)
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")


if __name__ == "__main__":
    orch = Orchestrator(notify=True)
    orch.run_daily()
